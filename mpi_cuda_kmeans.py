#!/usr/bin/env python
"""
MPI + CUDA 混合并行 K-Means 聚类实现（Python 版本）。

与同学实验配置保持一致：
- 默认 K = 8
- 最大迭代次数 = 30
- 收敛阈值 = 1e-4
- 随机种子 = 2025
- 本实验按照要求将 tm2002.img 复制 10 倍（--replicas=10）。

只对 K-Means 迭代阶段计时（compute_time_sec），不包含磁盘 IO。

运行示例（PowerShell）：
    mpiexec -n 8 python mpi_cuda_kmeans.py \
        --image ..\tm2002.img \
        --replicas 10 --clusters 8 --max-iter 30 \
        --chunk-size 100000 --metrics-out ..\metrics_mpi_cuda.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI

import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kmeans_common import (
    compute_local_bounds,
    evaluate_cluster_shift,
    iter_local_batches,
    load_image_pixels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPI + CUDA 的数据并行 K-Means")
    parser.add_argument(
        "--image",
        default=str(ROOT / "tm2002.img"),
        help="输入影像路径 (默认: 项目根目录下 tm2002.img)",
    )
    parser.add_argument("--replicas", type=int, default=10, help="数据复制次数")
    parser.add_argument("--clusters", type=int, default=8, help="聚类簇个数 K")
    parser.add_argument("--max-iter", type=int, default=30, help="最大迭代次数")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="聚类中心位移阈值")
    parser.add_argument("--chunk-size", type=int, default=400_000, help="GPU 批处理大小")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="显卡编号 (默认根据 rank 自动映射)",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=str(ROOT / "metrics_mpi_cuda.json"),
        help="保存性能指标的 JSON 文件",
    )
    parser.add_argument("--verbose", action="store_true", help="输出每轮迭代日志")
    return parser.parse_args()


def choose_initial_centers(
    base_pixels: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if k > base_pixels.shape[0]:
        raise ValueError(f"簇数 {k} 超过像元总数 {base_pixels.shape[0]}")
    selected = rng.choice(base_pixels.shape[0], size=k, replace=False)
    return base_pixels[selected].astype(np.float64)


def run_kmeans(args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, float]]:
    if args.max_iter <= 0:
        raise ValueError("max-iter 必须为正整数")
    if args.replicas <= 0:
        raise ValueError("replicas 必须为正整数")
    if args.clusters <= 0:
        raise ValueError("clusters 必须为正整数")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size 必须为正整数")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    total_start = MPI.Wtime()
    load_time = 0.0

    available_devices = cp.cuda.runtime.getDeviceCount()
    if available_devices == 0:
        raise RuntimeError("未检测到可用的 CUDA 设备")

    if args.device >= 0:
        device_id = args.device
    else:
        device_id = rank % available_devices

    if device_id >= available_devices:
        raise ValueError(
            f"rank {rank} 指定的 device {device_id} 超出范围 (0..{available_devices-1})"
        )

    cp.cuda.Device(device_id).use()

    if rank == 0:
        print(f"[MPI+CUDA] 开始加载影像: {args.image}", flush=True)
        load_begin = MPI.Wtime()
        base_pixels, summary = load_image_pixels(args.image)
        load_time = MPI.Wtime() - load_begin
        print(
            f"[MPI+CUDA] 影像加载完成: {summary.height}x{summary.width}, {summary.num_features} 波段, "
            f"{summary.num_pixels} 像元, 耗时 {load_time:.2f}s",
            flush=True,
        )
        print(
            f"[MPI+CUDA] 数据复制 {args.replicas} 次, 总像元数: {summary.num_pixels * args.replicas:,}",
            flush=True,
        )
        print(
            f"[MPI+CUDA] 开始 K-Means 聚类: K={args.clusters}, max_iter={args.max_iter}, "
            f"chunk_size={args.chunk_size}, ranks={world_size}",
            flush=True,
        )
        print(
            f"[MPI+CUDA] GPU 映射: {world_size} ranks → {available_devices} devices (rank {rank} → device {device_id})",
            flush=True,
        )
    else:
        base_pixels = None
        summary = None

    base_pixels = comm.bcast(base_pixels, root=0)
    summary = comm.bcast(summary, root=0)
    base_pixels = base_pixels.astype(np.float32, copy=False)

    num_features = base_pixels.shape[1]
    total_points = summary.num_pixels * args.replicas
    local_start, local_end = compute_local_bounds(total_points, world_size, rank)
    local_count = local_end - local_start

    rng = np.random.default_rng(args.seed)

    if rank == 0:
        centers = choose_initial_centers(base_pixels, args.clusters, rng)
    else:
        centers = np.empty((args.clusters, num_features), dtype=np.float64)
    comm.Bcast(centers, root=0)

    comm.Barrier()
    compute_start = MPI.Wtime()

    centers_gpu = cp.asarray(centers, dtype=cp.float32)
    final_inertia = 0.0
    iteration_counter = 0
    final_shift = 0.0
    global_counts = np.zeros(args.clusters, dtype=np.int64)
    global_sums = np.zeros((args.clusters, num_features), dtype=np.float64)

    total_batches = sum(1 for _ in iter_local_batches(local_start, local_end, args.chunk_size))

    for iteration in range(args.max_iter):
        iteration_counter = iteration + 1
        iter_start_time = MPI.Wtime()

        if args.verbose and rank == 0:
            print(f"\n[MPI+CUDA] 开始迭代 {iteration_counter}/{args.max_iter}...", flush=True)

        # 在 GPU 上累计每簇计数与加和，只在每轮结束时把聚合结果拷回 CPU
        local_counts_gpu = cp.zeros(args.clusters, dtype=cp.int64)
        local_sums_gpu = cp.zeros((args.clusters, num_features), dtype=cp.float32)
        local_inertia_gpu = cp.array(0.0, dtype=cp.float32)

        batch_idx = 0
        for batch_start, batch_end in iter_local_batches(
            local_start, local_end, args.chunk_size
        ):
            batch_idx += 1
            if args.verbose and rank == 0 and (
                batch_idx % max(1, total_batches // 10) == 0
                or batch_idx == total_batches
            ):
                progress = 100.0 * batch_idx / total_batches
                elapsed = MPI.Wtime() - iter_start_time
                rate = batch_idx / elapsed if elapsed > 0 else 0.0
                eta = (total_batches - batch_idx) / rate if rate > 0 else 0.0
                print(
                    f"  [进度] {progress:.1f}% ({batch_idx}/{total_batches} 批次, "
                    f"已用 {elapsed:.1f}s, 预计剩余 {eta:.1f}s)",
                    flush=True,
                )

            idx = np.arange(batch_start, batch_end, dtype=np.int64) % summary.num_pixels
            batch = base_pixels[idx].astype(np.float32, copy=False)

            batch_gpu = cp.asarray(batch, dtype=cp.float32)
            diff = batch_gpu[:, None, :] - centers_gpu[None, :, :]
            dist_sq = cp.sum(diff * diff, axis=2)
            labels_gpu = cp.argmin(dist_sq, axis=1)
            best_dist_sq = dist_sq[cp.arange(labels_gpu.size), labels_gpu]

            # 在 GPU 上直接累计惯量与每簇统计量
            local_inertia_gpu += best_dist_sq.sum()

            batch64_gpu = batch_gpu  # float32 足够，避免在消费级 GPU 上使用慢速 double
            counts_gpu = cp.bincount(labels_gpu, minlength=args.clusters)
            local_counts_gpu += counts_gpu

            for dim in range(num_features):
                weighted_gpu = cp.bincount(
                    labels_gpu, weights=batch64_gpu[:, dim], minlength=args.clusters
                )
                local_sums_gpu[:, dim] += weighted_gpu

            del batch_gpu, batch64_gpu, diff, dist_sq, labels_gpu, best_dist_sq

        cp.cuda.runtime.deviceSynchronize()

        # 每轮迭代结束时，仅把聚合后的结果拷回 CPU 再做 MPI 归约
        local_counts = cp.asnumpy(local_counts_gpu)
        local_sums = cp.asnumpy(local_sums_gpu).astype(np.float64, copy=False)
        local_inertia = float(local_inertia_gpu.get())

        global_counts.fill(0)
        global_sums.fill(0.0)
        comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
        comm.Allreduce(local_sums, global_sums, op=MPI.SUM)

        inertia_buf = np.array(local_inertia, dtype=np.float64)
        global_inertia = np.zeros_like(inertia_buf)
        comm.Allreduce(inertia_buf, global_inertia, op=MPI.SUM)
        final_inertia = float(global_inertia.item())

        if rank == 0:
            new_centers = np.divide(
                global_sums,
                global_counts[:, None],
                out=np.zeros_like(global_sums),
                where=global_counts[:, None] > 0,
            )
            empty_mask = global_counts == 0
            if np.any(empty_mask):
                replacements_idx = rng.integers(
                    0, base_pixels.shape[0], size=int(empty_mask.sum())
                )
                new_centers[empty_mask] = base_pixels[replacements_idx].astype(
                    np.float64
                )
        else:
            new_centers = np.empty_like(global_sums)

        comm.Bcast(new_centers, root=0)
        final_shift = evaluate_cluster_shift(centers, new_centers)
        centers = new_centers
        centers_gpu = cp.asarray(centers, dtype=cp.float32)

        if args.verbose and rank == 0:
            iter_time = MPI.Wtime() - iter_start_time
            print(
                f"[MPI+CUDA] iter={iteration_counter:02d} inertia={final_inertia:.3f} "
                f"shift={final_shift:.6f} non_empty={int(np.count_nonzero(global_counts))} "
                f"耗时={iter_time:.2f}s",
                flush=True,
            )

        if final_shift < args.tolerance:
            break

    comm.Barrier()
    compute_end = MPI.Wtime()
    total_time = MPI.Wtime() - total_start
    compute_time = compute_end - compute_start

    metrics: Dict[str, float] = {
        "method": "mpi+cuda",
        "image": str(Path(args.image).resolve()),
        "replicas": int(args.replicas),
        "clusters": int(args.clusters),
        "max_iter": int(args.max_iter),
        "actual_iters": int(iteration_counter),
        "tolerance": float(args.tolerance),
        "chunk_size": int(args.chunk_size),
        "seed": int(args.seed),
        "ranks": int(world_size),
        "device_id": int(device_id),
        "available_devices": int(available_devices),
        "pixels_per_rank": int(local_count),
        "total_pixels": int(total_points),
        "total_time_sec": float(total_time),
        "compute_time_sec": float(compute_time),
        "io_time_sec": float(load_time),
        "final_inertia": float(final_inertia),
        "final_shift": float(final_shift),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_height": int(summary.height),
        "image_width": int(summary.width),
        "num_features": int(summary.num_features),
    }

    extras = {
        "global_counts": None,
        "final_centers": None,
    }

    if rank == 0:
        extras["global_counts"] = global_counts.tolist()
        extras["final_centers"] = centers.tolist()

    return metrics, extras


def main() -> None:
    args = parse_args()
    metrics, extras = run_kmeans(args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print(
            f"[MPI+CUDA] 总耗时 {metrics['total_time_sec']:.3f}s | "
            f"迭代耗时 {metrics['compute_time_sec']:.3f}s | "
            f"迭代次数 {metrics['actual_iters']} | "
            f"最终惯量 {metrics['final_inertia']:.3f}",
            flush=True,
        )

        if args.metrics_out:
            out_path = Path(args.metrics_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "metrics": metrics,
                        "global_counts": extras["global_counts"],
                        "final_centers": extras["final_centers"],
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[MPI+CUDA] 指标已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
