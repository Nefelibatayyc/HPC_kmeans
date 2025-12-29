#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

// -------------------- 参数 --------------------
const int K = 8;
const int max_iter = 30;
const float epsilon = 1e-4f;
const int REPLICATE = 10;

// -------------------- 距离计算 --------------------
float dist2(const vector<float>& a, const vector<float>& b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

// -------------------- MPI K-Means --------------------
void kmeans_mpi(const vector<vector<float>>& data, long long N, int bandCount,
    vector<int>& labels, int& iterations, vector<float>& finalMaxChange,
    double& totalAssignmentTime, double& totalUpdateTime)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N_local = N / size;
    long long start = rank * N_local;
    long long end = (rank == size - 1) ? N : start + N_local;

    if (rank == 0) labels.resize(N);  // 只有 rank0 保存全部标签（后面只用前 original_pixels）

    // 初始化质心（所有进程必须完全相同）
    vector<vector<float>> centroids(K, vector<float>(bandCount));
    if (rank == 0) {
        srand(12345);
        for (int k = 0; k < K; ++k) {
            centroids[k] = data[rand() % N];
        }
    }
    // 广播初始质心
    for (int k = 0; k < K; ++k) {
        MPI_Bcast(centroids[k].data(), bandCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    totalAssignmentTime = 0.0;
    totalUpdateTime = 0.0;

    vector<float> local_sum(K * bandCount, 0.0f);
    vector<int>   local_count(K, 0);

    for (int iter = 0; iter < max_iter; ++iter) {
        if (rank == 0) iterations = iter + 1;

        auto t1 = high_resolution_clock::now();

        // =============== 本地 Assignment ===============
        for (long long i = start; i < end; ++i) {
            float min_dist = 1e10f;
            int best_k = 0;
            for (int k = 0; k < K; ++k) {
                float d = dist2(data[i], centroids[k]);
                if (d < min_dist) {
                    min_dist = d;
                    best_k = k;
                }
            }
            if (rank == 0) labels[i] = best_k;  // 只在 rank0 保存标签
        }

        auto t2 = high_resolution_clock::now();
        totalAssignmentTime += duration<double>(t2 - t1).count();

        // =============== 本地统计 sum 和 count ===============
        fill(local_sum.begin(), local_sum.end(), 0.0f);
        fill(local_count.begin(), local_count.end(), 0);

        for (long long i = start; i < end; ++i) {
            int best_k = 0;
            float min_dist = 1e10f;
            for (int k = 0; k < K; ++k) {
                float d = dist2(data[i], centroids[k]);
                if (d < min_dist) {
                    min_dist = d;
                    best_k = k;
                }
            }
            local_count[best_k]++;
            for (int b = 0; b < bandCount; ++b) {
                local_sum[best_k * bandCount + b] += data[i][b];
            }
        }

        auto t3 = high_resolution_clock::now();

        // =============== 全局归约 ===============
        vector<float> global_sum(K * bandCount, 0.0f);
        vector<int>   global_count(K, 0);

        MPI_Allreduce(local_sum.data(), global_sum.data(), K * bandCount,
            MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_count.data(), global_count.data(), K,
            MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // =============== 更新质心（所有进程同步更新） ===============
        float max_change = 0.0f;
        for (int k = 0; k < K; ++k) {
            if (global_count[k] == 0) continue;
            for (int b = 0; b < bandCount; ++b) {
                float new_c = global_sum[k * bandCount + b] / global_count[k];
                max_change = max(max_change, fabsf(new_c - centroids[k][b]));
                centroids[k][b] = new_c;
            }
        }

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
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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