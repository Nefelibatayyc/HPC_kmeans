#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace std::chrono;

// -------------------- 参数 --------------------
const int K = 8;
const int max_iter = 30;
const float epsilon = 1e-4f;
const int REPLICATE = 10;

// ==================== 新增：CUDA Kernel（只加这一个函数）================
__global__ void cuda_kmeans_assign(
    const float* data,          // [N_local * bandCount]
    const float* centroids,     // [K * bandCount]
    int* labels,                // [N_local]
    long long N_local,
    int bandCount)
{
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_local) return;

    float min_dist = 1e10f;
    int   best_k = 0;

    for (int k = 0; k < K; ++k) {
        float dist = 0.0f;
        #pragma unroll
        for (int b = 0; b < bandCount; ++b) {
            float diff = data[i * bandCount + b] - centroids[k * bandCount + b];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_k = k;
        }
    }
    labels[i] = best_k;
}

// -------------------- 距离计算（CPU 版保留，仅作备用） --------------------
float dist2(const vector<float>& a, const vector<float>& b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

// -------------------- MPI + CUDA K-Means --------------------
extern "C" void kmeans_mpi(const vector<vector<float>>& data, long long N, int bandCount,
    vector<int>& labels, int& iterations, vector<float>& finalMaxChange,
    double& totalAssignmentTime, double& totalUpdateTime)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N_local = N / size;
    long long start = rank * N_local;
    long long end = (rank == size - 1) ? N : start + N_local;
    N_local = end - start;  // 防止末尾进程多几个

    if (rank == 0) labels.resize(N);

    // ==================== 新增：GPU 内存分配（只在这里加 6 行）================
    float* d_data = nullptr, * d_centroids = nullptr;
    int* d_labels = nullptr;

    cudaMalloc(&d_data, N_local * bandCount * sizeof(float));
    cudaMalloc(&d_centroids, K * bandCount * sizeof(float));
    cudaMalloc(&d_labels, N_local * sizeof(int));

    // 把本地数据一次性拷到 GPU（只拷一次，后面一直用）
    vector<float> local_flat(N_local * bandCount);
    for (long long i = 0; i < N_local; ++i) {
        for (int b = 0; b < bandCount; ++b) {
            local_flat[i * bandCount + b] = data[start + i][b];
        }
    }
    cudaMemcpy(d_data, local_flat.data(), N_local * bandCount * sizeof(float), cudaMemcpyHostToDevice);

    // 初始化质心
    vector<float> centroids_flat(K * bandCount, 0.0f);
    if (rank == 0) {
        srand(12345);
        for (int k = 0; k < K; ++k) {
            long long idx = rand() % N;
            for (int b = 0; b < bandCount; ++b) {
                centroids_flat[k * bandCount + b] = data[idx][b];
            }
        }
    }
    MPI_Bcast(centroids_flat.data(), K * bandCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
    cudaMemcpy(d_centroids, centroids_flat.data(), K * bandCount * sizeof(float), cudaMemcpyHostToDevice);

    totalAssignmentTime = 0.0;
    totalUpdateTime = 0.0;

    vector<float> local_sum(K * bandCount, 0.0f);
    vector<int>   local_count(K, 0);

    for (int iter = 0; iter < max_iter; ++iter) {
        if (rank == 0) iterations = iter + 1;

        auto t1 = high_resolution_clock::now();

        // ==================== 关键修改：GPU 完成 Assignment ====================
        dim3 block(256);
        dim3 grid((N_local + block.x - 1) / block.x);
        //cuda_kmeans_assign << <grid, block >> > (d_data, d_centroids, d_labels, N_local, bandCount);
        cudaDeviceSynchronize();

        auto t2 = high_resolution_clock::now();
        totalAssignmentTime += duration<double>(t2 - t1).count();

        // 把标签拷回 CPU 用于归约
        vector<int> h_labels(N_local);
        cudaMemcpy(h_labels.data(), d_labels, N_local * sizeof(int), cudaMemcpyDeviceToHost);

        // ==================== 下面从统计到结束，和你原代码完全一样 ====================
        fill(local_sum.begin(), local_sum.end(), 0.0f);
        fill(local_count.begin(), local_count.end(), 0);

        for (long long i = 0; i < N_local; ++i) {
            int best_k = h_labels[i];
            local_count[best_k]++;
            for (int b = 0; b < bandCount; ++b) {
                local_sum[best_k * bandCount + b] += data[start + i][b];
            }
        }

        auto t3 = high_resolution_clock::now();

        vector<float> global_sum(K * bandCount, 0.0f);
        vector<int>   global_count(K, 0);
        MPI_Allreduce(local_sum.data(), global_sum.data(), K * bandCount,
            MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_count.data(), global_count.data(), K,
            MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        float max_change = 0.0f;
        for (int k = 0; k < K; ++k) {
            if (global_count[k] == 0) continue;
            for (int b = 0; b < bandCount; ++b) {
                float new_c = global_sum[k * bandCount + b] / global_count[k];
                max_change = max(max_change, fabsf(new_c - centroids_flat[k * bandCount + b]));
                centroids_flat[k * bandCount + b] = new_c;
            }
        }
        // 拷回 GPU
        cudaMemcpy(d_centroids, centroids_flat.data(), K * bandCount * sizeof(float), cudaMemcpyHostToDevice);

        if (rank == 0) {
            finalMaxChange.push_back(max_change);
        }

        auto t4 = high_resolution_clock::now();
        totalUpdateTime += duration<double>(t4 - t3).count();

        if (max_change < epsilon) {
            if (rank == 0) iterations = iter + 1;
            break;
        }
    }

    // === 在释放 GPU 之前，拷回并收集 labels 到 root ===
// 从 device 拷回本地最终标签
    vector<int> final_h_labels(N_local);
    cudaMemcpy(final_h_labels.data(), d_labels, N_local * sizeof(int), cudaMemcpyDeviceToHost);

    // 每个进程的本地长度（保证能用 int，若超 int 需改用分块或其他方式）
    int sendcount = static_cast<int>(N_local);

    // root 先收集每个进程的 sendcount
    vector<int> recvcounts;
    if (rank == 0) recvcounts.resize(size);
    MPI_Gather(&sendcount, 1, MPI_INT,
        rank == 0 ? recvcounts.data() : nullptr,
        1, MPI_INT, 0, MPI_COMM_WORLD);

    // root 计算 displacements
    vector<int> displs;
    if (rank == 0) {
        displs.resize(size);
        displs[0] = 0;
        for (int r = 1; r < size; ++r)
            displs[r] = displs[r - 1] + recvcounts[r - 1];
    }

    // 把各进程的 final_h_labels 收集到 root 的 labels（labels 已在 root 预先 resize(N)）
    MPI_Gatherv(final_h_labels.data(), sendcount, MPI_INT,
        rank == 0 ? labels.data() : nullptr,
        rank == 0 ? recvcounts.data() : nullptr,
        rank == 0 ? displs.data() : nullptr,
        MPI_INT, 0, MPI_COMM_WORLD);


    // ==================== 新增：释放 GPU 内存 ====================
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
}

// ==================== main 函数完全不动！================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---------- CUDA 初始化（自动选卡） ----------
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaSetDevice(rank % deviceCount);
    }
    else if (rank == 0) {
        cout << "Warning: No GPU found, running on CPU only.\n";
    }

    auto t_program_start = high_resolution_clock::now();

    if (rank == 0) {
        GDALAllRegister();
        cout << "=== MPI K-Means (processes: " << size << ") ===\n";
    }

    const char* filename = "data/tm2002.img";
    GDALDataset* dataset = nullptr;
    int width = 0, height = 0, bandCount = 0;
    long long original_pixels = 0;

    if (rank == 0) {
        dataset = (GDALDataset*)GDALOpen(filename, GA_ReadOnly);
        if (!dataset) {
            cerr << "Rank 0: Cannot open file: " << filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        width = dataset->GetRasterXSize();
        height = dataset->GetRasterYSize();
        bandCount = dataset->GetRasterCount();
        original_pixels = (long long)width * height;

        cout << "Original image: " << width << " x " << height
            << ", bands: " << bandCount << endl;
    }

    // 广播影像基本信息
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bandCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&original_pixels, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    long long total_pixels = original_pixels * REPLICATE;

    if (rank == 0) {
        cout << "Replicating data " << REPLICATE << " times → "
            << total_pixels << " pixels (" << total_pixels / 1e6 << " M)" << endl;
    }

    // ==================== 读取 + 复制数据（每个进程独立一份完整数据） ====================
    vector<vector<float>> data(total_pixels, vector<float>(bandCount));

    if (rank == 0) {
        // rank0 先读原始影像
        vector<vector<float>> raw_data(original_pixels, vector<float>(bandCount));
        for (int b = 0; b < bandCount; ++b) {
            GDALRasterBand* band = dataset->GetRasterBand(b + 1);
            vector<float> buf(original_pixels);
            band->RasterIO(GF_Read, 0, 0, width, height, buf.data(), width, height,
                GDT_Float32, 0, 0);
            for (long long i = 0; i < original_pixels; ++i) {
                raw_data[i][b] = buf[i];
            }
        }

        // 复制 REPLICATE 份
        for (int r = 0; r < REPLICATE; ++r) {
            size_t offset = (size_t)r * original_pixels;
            for (long long i = 0; i < original_pixels; ++i) {
                data[offset + i] = raw_data[i];
            }
        }
        GDALClose(dataset);
    }

    // 所有进程广播完整 data（你数据只有 174 MB，广播完全可接受）
    for (long long i = 0; i < total_pixels; ++i) {
        MPI_Bcast(data[i].data(), bandCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // ==================== 执行 MPI K-Means ====================
    vector<int>    labels;
    vector<float>  finalMaxChange;
    int            iterations = 0;
    double         assignTime = 0.0, updateTime = 0.0;

    if (rank == 0) {
        cout << "Starting MPI K-Means on " << size << " processes..." << endl;
    }

    auto t_kmeans_start = high_resolution_clock::now();

    kmeans_mpi(data, total_pixels, bandCount, labels,
        iterations, finalMaxChange, assignTime, updateTime);

    auto t_kmeans_end = high_resolution_clock::now();
    double kmeansTime = duration<double>(t_kmeans_end - t_kmeans_start).count();

    if (rank == 0) {
        float lastChange = finalMaxChange.empty() ? 0.0f : finalMaxChange.back();

        cout << "=============================================" << endl;
        cout << "MPI K-Means finished! (np = " << size << ")" << endl;
        cout << "Iterations            : " << iterations << endl;
        cout << "Last max_change       : " << lastChange << endl;
        cout << "Pure K-Means time     : " << fixed << kmeansTime << " s" << endl;
        cout << "=============================================" << endl;

        // ==================== 输出结果（只输出第一份） ====================
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        const char* outFile = "data/tm2002_kmeans_mpi.tif";
        GDALDataset* outDs = driver->Create(outFile, width, height, 1, GDT_Byte, nullptr);

        double gt[6];
        dataset = (GDALDataset*)GDALOpen(filename, GA_ReadOnly);  // 重新打开拿投影
        if (dataset->GetGeoTransform(gt) == CE_None) outDs->SetGeoTransform(gt);
        const char* prj = dataset->GetProjectionRef();
        if (prj && strlen(prj)) outDs->SetProjection(prj);
        GDALClose(dataset);

        vector<unsigned char> buf(original_pixels);
        for (long long i = 0; i < original_pixels; ++i) {
            buf[i] = static_cast<unsigned char>(labels[i]);
        }
        outDs->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height,
            buf.data(), width, height, GDT_Byte, 0, 0);
        GDALClose(outDs);

        auto t_end = high_resolution_clock::now();
        cout << "Total program time    : " << duration<double>(t_end - t_program_start).count() << " s" << endl;
        cout << "MPI result           → " << outFile << endl;
    }

    MPI_Finalize();
    return 0;
}