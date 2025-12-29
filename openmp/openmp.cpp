#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <omp.h>                     // <--- 新增

using namespace std;
using namespace std::chrono;

// -------------------- 参数 --------------------
const int K = 8;                // 聚类类别数
const int max_iter = 30;       // 最大迭代次数
const float epsilon = 1e-4;     // 收敛阈值
const int REPLICATE = 10;        // 数据放大倍数
const int NUM_THREADS = 8;

// -------------------- 距离计算 --------------------
float dist2(const vector<float>& a, const vector<float>& b) {
    float d = d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

// -------------------- OpenMP 并行 K-Means --------------------
void kmeans_omp(const vector<vector<float>>& data, long long N, int bandCount,
    vector<int>& labels,
    int& iterations, vector<float>& finalMaxChange,
    double& totalAssignmentTime, double& totalUpdateTime)
{
    labels.resize(N);

    // 初始化质心（保持和串行版完全相同的随机种子）
    vector<vector<float>> centroids(K, vector<float>(bandCount));
    srand(12345);
    for (int k = 0; k < K; ++k) {
        centroids[k] = data[rand() % N];
    }

    totalAssignmentTime = 0.0;
    totalUpdateTime = 0.0;

    // 为每个线程准备私有临时数组（避免频繁分配）
    int max_threads = NUM_THREADS;
    vector<vector<vector<float>>> thread_sums(max_threads,
        vector<vector<float>>(K, vector<float>(bandCount, 0.0f)));
    vector<vector<int>>           thread_counts(max_threads, vector<int>(K, 0));

    for (int iter = 0; iter < max_iter; ++iter) {
        iterations = iter + 1;

        // ==================== 分配阶段（最耗时） ====================
        auto t_assign_start = high_resolution_clock::now();

#pragma omp parallel for schedule(static)
        for (long long i = 0; i < N; ++i) {
            float min_dist = 1e10f;
            int   best_k = 0;

            for (int k = 0; k < K; ++k) {
                float d = dist2(data[i], centroids[k]);
                if (d < min_dist) {
                    min_dist = d;
                    best_k = k;
                }
            }
            labels[i] = best_k;
        }

        auto t_assign_end = high_resolution_clock::now();
        totalAssignmentTime += duration<double>(t_assign_end - t_assign_start).count();

        //// ==================== 更新质心阶段（归约） ====================
        //auto t_update_start = high_resolution_clock::now();

        //// 1) 先让每个线程统计自己的局部和与计数
        //#pragma omp parallel
        //{
        //    int tid = omp_get_thread_num();
        //    auto& local_sum = thread_sums[tid];
        //    auto& local_count = thread_counts[tid];

        //    // 清零
        //    for (int k = 0; k < K; ++k) {
        //        local_count[k] = 0;
        //        for (int b = 0; b < bandCount; ++b) {
        //            local_sum[k][b] = 0.0f;
        //        }
        //    }

        //    // 累加
        //    #pragma omp for schedule(static) nowait
        //    for (long long i = 0; i < N; ++i) {
        //        int k = labels[i];
        //        local_count[k]++;
        //        for (int b = 0; b < bandCount; ++b) {
        //            local_sum[k][b] += data[i][b];
        //        }
        //    }

        //    // 2) 主线程把所有线程的结果归约到全局
        //    #pragma omp barrier
        //    #pragma omp master
        //    {
        //        vector<vector<float>> global_sum(K, vector<float>(bandCount, 0.0f));
        //        vector<int>           global_count(K, 0);

        //        for (int t = 0; t < max_threads; ++t) {
        //            for (int k = 0; k < K; ++k) {
        //                global_count[k] += thread_counts[t][k];
        //                for (int b = 0; b < bandCount; ++b) {
        //                    global_sum[k][b] += thread_sums[t][k][b];
        //                }
        //            }
        //        }

        //        // 计算新质心并计算最大移动距离
        //        float max_change = 0.0f;
        //        for (int k = 0; k < K; ++k) {
        //            if (global_count[k] == 0) continue;
        //            for (int b = 0; b < bandCount; ++b) {
        //                float new_c = global_sum[k][b] / global_count[k];
        //                max_change = max(max_change, fabsf(new_c - centroids[k][b]));
        //                centroids[k][b] = new_c;
        //            }
        //        }
        //        finalMaxChange.push_back(max_change);

        //        if (max_change < epsilon) {
        //            // 通知所有线程提前结束外层 for 循环（通过环境变量或标志均可，这里直接 break）
        //            iter = max_iter;  // 强制跳出
        //        }
        //    }
        //}  // end parallel

        //auto t_update_end = high_resolution_clock::now();
        //totalUpdateTime += duration<double>(t_update_end - t_update_start).count();

        //if (finalMaxChange.back() < epsilon) break;

        auto t_update_start = high_resolution_clock::now();

        // 直接在 parallel for 中做 reduction，OpenMP 自动优化，几乎零开销
        vector<vector<float>> sum(K, vector<float>(bandCount, 0.0f));
        vector<int>           count(K, 0);

#pragma omp parallel for reduction(+:sum[:K][:bandCount], count[:K]) schedule(static)
        for (long long i = 0; i < N; ++i) {
            int k = labels[i];
            count[k]++;
            for (int b = 0; b < bandCount; ++b) {
                sum[k][b] += data[i][b];
            }
        }

        // 主线程计算新质心和最大变化量
        float max_change = 0.0f;
        for (int k = 0; k < K; ++k) {
            if (count[k] == 0) continue;
            for (int b = 0; b < bandCount; ++b) {
                float new_centroid = sum[k][b] / count[k];
                max_change = max(max_change, fabsf(new_centroid - centroids[k][b]));
                centroids[k][b] = new_centroid;
            }
        }

        finalMaxChange.push_back(max_change);
        auto t_update_end = high_resolution_clock::now();
        totalUpdateTime += duration<double>(t_update_end - t_update_start).count();

        if (max_change < epsilon) {
            break;  // 可以正常提前退出
        }
    }
}

int main() {
    GDALAllRegister();

    omp_set_num_threads(NUM_THREADS);

    auto t_program_start = high_resolution_clock::now();

    const char* filename = "data/tm2002.img";
    GDALDataset* dataset = (GDALDataset*)GDALOpen(filename, GA_ReadOnly);
    if (!dataset) {
        cerr << "Cannot open file: " << filename << endl;
        return -1;
    }

    int width = dataset->GetRasterXSize();
    int height = dataset->GetRasterYSize();
    int bandCount = dataset->GetRasterCount();

    cout << "Original image: " << width << " x " << height
        << ", bands: " << bandCount << endl;

    // ==================== 1. 读取原始数据 ====================
    long long original_pixels = (long long)width * height;
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

    // ==================== 2. 数据放大 ====================
    long long total_pixels = original_pixels * REPLICATE;

    cout << "Replicating data " << REPLICATE << " times → "
        << total_pixels << " pixels (" << total_pixels / 1e6 << " M)" << endl;

    vector<vector<float>> data(total_pixels, vector<float>(bandCount));

    for (int r = 0; r < REPLICATE; ++r) {
        size_t offset = (size_t)r * original_pixels;
        for (long long i = 0; i < original_pixels; ++i) {
            data[offset + i] = raw_data[i];
        }
    }

    // ==================== 3. OpenMP K-Means ====================
    vector<int>    labels(total_pixels);
    vector<float>  finalMaxChange;
    int            iterations = 0;
    double         assignTime = 0.0, updateTime = 0.0;

    cout << "Starting OpenMP K-Means on " << REPLICATE << "x data (threads="
        << NUM_THREADS << ")..." << endl;

    auto t_kmeans_start = high_resolution_clock::now();

    kmeans_omp(data, total_pixels, bandCount, labels,
        iterations, finalMaxChange, assignTime, updateTime);

    auto t_kmeans_end = high_resolution_clock::now();
    double kmeansTime = duration<double>(t_kmeans_end - t_kmeans_start).count();

    float lastMaxChange = finalMaxChange.empty() ? 0.0f : finalMaxChange.back();

    cout << "=============================================" << endl;
    cout << "OpenMP K-Means finished!" << endl;
    cout << "Threads               : " << NUM_THREADS << endl;
    cout << "Iterations            : " << iterations << endl;
    cout << "Last max_change       : " << lastMaxChange << endl;
    cout << "Pure K-Means time     : " << fixed << kmeansTime << " s" << endl;
    cout << "=============================================" << endl;

    // ==================== 4. 输出结果（与 cpu.cpp 完全一致） ====================
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) { cerr << "GTiff driver not available!" << endl; GDALClose(dataset); return -1; }

    const char* outFile = "data/tm2002_kmeans_openmp.tif";

    GDALDataset* outDs = driver->Create(outFile, width, height, 1, GDT_Byte, nullptr);
    if (!outDs) { cerr << "Cannot create " << outFile << endl; GDALClose(dataset); return -1; }

    double gt[6];
    if (dataset->GetGeoTransform(gt) == CE_None) outDs->SetGeoTransform(gt);
    const char* prj = dataset->GetProjectionRef();
    if (prj && strlen(prj)) outDs->SetProjection(prj);

    vector<unsigned char> buf(width * height);
    for (long long i = 0; i < original_pixels; ++i) {
        buf[i] = static_cast<unsigned char>(labels[i]);
    }
    outDs->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height,
        buf.data(), width, height, GDT_Byte, 0, 0);

    GDALClose(outDs);
    GDALClose(dataset);

    auto t_program_end = high_resolution_clock::now();
    double totalTime = duration<double>(t_program_end - t_program_start).count();
    cout << "Total program time (including IO): " << totalTime << " s" << endl;
    cout << "OpenMP result → " << outFile << endl;

    return 0;
}