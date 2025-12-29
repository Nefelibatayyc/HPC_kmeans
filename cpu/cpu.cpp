#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <gdal_priv.h>c
#include <cpl_conv.h>

using namespace std;
using namespace std::chrono;

// -------------------- 参数 --------------------
const int K = 8;                // 聚类类别数
const int max_iter = 30;       // 最大迭代次数
const float epsilon = 1e-4;     // 收敛阈值
const int REPLICATE = 20;       // 数据放大倍数

// -------------------- 距离计算 --------------------
float dist2(const vector<float>& a, const vector<float>& b) {
    float d = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}
// -------------------- K-Means --------------------
void kmeans_cpu(const vector<vector<float>>& data, int N, int bandCount,
    vector<int>& labels,
    int& iterations, vector<float>& finalMaxChange,
    double& totalAssignmentTime, double& totalUpdateTime) {
    labels.resize(N);
    // 初始化质心
    vector<vector<float>> centroids(K, vector<float>(bandCount));
    srand(12345);
    for (int k = 0; k < K; k++) {
        centroids[k] = data[rand() % N];
    }
    totalAssignmentTime = 0.0;
    totalUpdateTime = 0.0;
    for (int iter = 0; iter < max_iter; iter++) {
        iterations = iter + 1;
        auto t_assign_start = high_resolution_clock::now();
        // -------- 分配阶段 --------
        for (int i = 0; i < N; i++) {
            float min_dist = 1e10;
            int best_k = 0;
            for (int k = 0; k < K; k++) {
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
        auto t_update_start = high_resolution_clock::now();
        // -------- 更新质心阶段 --------
        vector<vector<float>> sum(K, vector<float>(bandCount, 0.0));
        vector<int> count(K, 0);
        for (int i = 0; i < N; i++) {
            int k = labels[i];
            for (int b = 0; b < bandCount; b++)
                sum[k][b] += data[i][b];
            count[k]++;
        }
        float max_change = 0.0;
        for (int k = 0; k < K; k++) {
            if (count[k] == 0) continue;
            for (int b = 0; b < bandCount; b++) {
                float new_centroid = sum[k][b] / count[k];
                max_change = max(max_change, fabs(new_centroid - centroids[k][b]));
                centroids[k][b] = new_centroid;
            }
        }
        auto t_update_end = high_resolution_clock::now();
        totalUpdateTime += duration<double>(t_update_end - t_update_start).count();
        finalMaxChange.push_back(max_change);
        if (max_change < epsilon) break;
    }
}

int main() {
    GDALAllRegister();

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

    // ==================== 1. 只读一次原始数据 ====================
    vector<vector<float>> raw_data(width * height, vector<float>(bandCount));
    for (int b = 0; b < bandCount; ++b) {
        GDALRasterBand* band = dataset->GetRasterBand(b + 1);
        vector<float> buf(width * height);
        band->RasterIO(GF_Read, 0, 0, width, height, buf.data(), width, height, GDT_Float32, 0, 0);
        for (int i = 0; i < width * height; ++i) {
            raw_data[i][b] = buf[i];
        }
    }

    // ==================== 2. 数据放大倍数 ====================
    long long original_pixels = (long long)width * height;
    long long total_pixels = original_pixels * REPLICATE;

    cout << "Replicating data " << REPLICATE << " times → "
        << total_pixels << " pixels (" << total_pixels / 1e6 << " M)" << endl;

    vector<vector<float>> data(total_pixels, vector<float>(bandCount));

    for (int r = 0; r < REPLICATE; ++r) {
        size_t offset = (size_t)r * original_pixels;
        for (long long i = 0; i < original_pixels; ++i) {
            data[offset + i] = raw_data[i];   // 直接拷贝整行向量，很快
        }
    }

    // ==================== 3. K-Means 聚类（只计纯计算时间） ====================
    vector<int>    labels(total_pixels);               // 注意大小也要 ×REPLICATE
    vector<float>  finalMaxChange;
    int            iterations = 0;
    double         assignTime = 0.0, updateTime = 0.0;

    cout << "Starting K-Means on " << REPLICATE << "x data..." << endl;
    auto t_kmeans_start = high_resolution_clock::now();

    kmeans_cpu(data, total_pixels, bandCount, labels,
        iterations, finalMaxChange, assignTime, updateTime);

    auto t_kmeans_end = high_resolution_clock::now();
    double kmeansTime = duration<double>(t_kmeans_end - t_kmeans_start).count();

    float lastMaxChange = finalMaxChange.empty() ? 0.0f : finalMaxChange.back();

    cout << "=============================================" << endl;
    cout << "K-Means finished!" << endl;
    cout << "Iterations            : " << iterations << endl;
    cout << "Last max_change       : " << lastMaxChange << endl;
    cout << "Pure K-Means time     : " << fixed << kmeansTime << " s" << endl;
    cout << "=============================================" << endl;

    // ==================== 4. 输出结果 ====================

    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) { cerr << "GTiff driver not available!" << endl; GDALClose(dataset); return -1; }

    const char* outFileA = "data/tm2002_kmeans_cpu.tif";

    GDALDataset* outDsA = driver->Create(outFileA, width, height, 1, GDT_Byte, nullptr);
    if (!outDsA) { cerr << "Cannot create " << outFileA << endl; GDALClose(dataset); return -1; }

    // 复制地理信息
    double gt[6];
    if (dataset->GetGeoTransform(gt) == CE_None) outDsA->SetGeoTransform(gt);
    const char* prj = dataset->GetProjectionRef();
    if (prj && strlen(prj)) outDsA->SetProjection(prj);

    // 只写第 0 份的第一份结果（和其他代码完全兼容）
    vector<unsigned char> bufA(width * height);
    for (long long i = 0; i < original_pixels; ++i) {
        bufA[i] = static_cast<unsigned char>(labels[i]);
    }
    outDsA->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height,
        bufA.data(), width, height, GDT_Byte, 0, 0);

    GDALClose(outDsA);
    GDALClose(dataset);

    auto t_program_end = high_resolution_clock::now();
    double totalTime = duration<double>(t_program_end - t_program_start).count();
    cout << "Total program time (including IO): " << totalTime << " s" << endl;
    cout << "Normal result    → " << outFileA << endl;

    return 0;
}


//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <cstdlib>
//#include <ctime>
//#include <chrono>
//#include <gdal_priv.h>
//#include <cpl_conv.h>
//
//using namespace std;
//using namespace std::chrono;
//
//// -------------------- ���� --------------------
//const int K = 8;
//const int max_iter = 1000;
//const int REPLICATE = 30;
//const float epsilon = 1e-4;
//
//// -------------------- ������� --------------------
//float dist2(const vector<float>& a, const vector<float>& b) {
//    float d = 0.0;
//    for (size_t i = 0; i < a.size(); ++i) {
//        float diff = a[i] - b[i];
//        d += diff * diff;
//    }
//    return d;
//}
//
//// -------------------- K-Means (���ֲ���) --------------------
//void kmeans_cpu(const vector<vector<float>>& data, int N, int bandCount,
//    vector<int>& labels,
//    int& iterations, vector<float>& finalMaxChange,
//    double& totalAssignmentTime, double& totalUpdateTime) {
//    labels.resize(N);
//    // ��ʼ������
//    vector<vector<float>> centroids(K, vector<float>(bandCount));
//    srand(12345);
//    for (int k = 0; k < K; k++) {
//        centroids[k] = data[rand() % N];
//    }
//    totalAssignmentTime = 0.0;
//    totalUpdateTime = 0.0;
//    for (int iter = 0; iter < max_iter; iter++) {
//        iterations = iter + 1;
//        auto t_assign_start = high_resolution_clock::now();
//        // -------- ����׶� --------
//        for (int i = 0; i < N; i++) {
//            float min_dist = 1e10;
//            int best_k = 0;
//            for (int k = 0; k < K; k++) {
//                float d = dist2(data[i], centroids[k]);
//                if (d < min_dist) {
//                    min_dist = d;
//                    best_k = k;
//                }
//            }
//            labels[i] = best_k;
//        }
//        auto t_assign_end = high_resolution_clock::now();
//        totalAssignmentTime += duration<double>(t_assign_end - t_assign_start).count();
//        auto t_update_start = high_resolution_clock::now();
//        // -------- �������Ľ׶� --------
//        vector<vector<float>> sum(K, vector<float>(bandCount, 0.0));
//        vector<int> count(K, 0);
//        for (int i = 0; i < N; i++) {
//            int k = labels[i];
//            for (int b = 0; b < bandCount; b++)
//                sum[k][b] += data[i][b];
//            count[k]++;
//        }
//        float max_change = 0.0;
//        for (int k = 0; k < K; k++) {
//            if (count[k] == 0) continue;
//            for (int b = 0; b < bandCount; b++) {
//                float new_centroid = sum[k][b] / count[k];
//                max_change = max(max_change, fabs(new_centroid - centroids[k][b]));
//                centroids[k][b] = new_centroid;
//            }
//        }
//        auto t_update_end = high_resolution_clock::now();
//        totalUpdateTime += duration<double>(t_update_end - t_update_start).count();
//        finalMaxChange.push_back(max_change);
//        if (max_change < epsilon) break;
//    }
//}
//
//// -------------------- ���������ؼ��޸Ĳ��֣� --------------------
//int main() {
//    GDALAllRegister();
//
//    auto t_start = high_resolution_clock::now();
//    const char* filename = "data/tm2002.img";
//    GDALDataset* dataset = (GDALDataset*)GDALOpen(filename, GA_ReadOnly);
//    if (!dataset) {
//        cerr << "Cannot open file: " << filename << endl;
//        return -1;
//    }
//
//    int width = dataset->GetRasterXSize();
//    int height = dataset->GetRasterYSize();
//    int bandCount = dataset->GetRasterCount();
//
//    cout << "Image size: " << width << " x " << height << ", bands: " << bandCount << endl;
//
//    // ======== ��ȡ���в��� ========
//    vector<vector<float>> raw_data(width * height, vector<float>(bandCount));
//    for (int b = 0; b < bandCount; ++b) {
//        GDALRasterBand* band = dataset->GetRasterBand(b + 1);
//        vector<float> buf(width * height);
//        band->RasterIO(GF_Read, 0, 0, width, height, buf.data(), width, height, GDT_Float32, 0, 0);
//        for (int i = 0; i < width * height; ++i) {
//            raw_data[i][b] = buf[i];
//        }
//    }
//
//    long long original_pixels = (long long)width * height;
//    long long total_pixels = original_pixels * REPLICATE;
//
//    cout << "Replicating data " << REPLICATE << " times → "
//        << total_pixels << " pixels (" << total_pixels / 1e6 << " M)" << endl;
//
//    vector<vector<float>> data(total_pixels, vector<float>(bandCount));
//
//    for (int r = 0; r < REPLICATE; ++r) {
//        size_t offset = (size_t)r * original_pixels;
//        for (long long i = 0; i < original_pixels; ++i) {
//            data[offset + i] = raw_data[i];   // ֱ�ӿ��������������ܿ�
//        }
//    }
//
//    // ======== K-Means ���� ========
//    vector<int> labels;
//    vector<float> finalMaxChange;
//    int iterations = 0;
//    double totalAssignmentTime = 0.0, totalUpdateTime = 0.0;
//
//    auto t_kmeans_start = high_resolution_clock::now();
//    kmeans_cpu(data, width * height, bandCount, labels,
//        iterations, finalMaxChange, totalAssignmentTime, totalUpdateTime);
//    auto t_kmeans_end = high_resolution_clock::now();
//    double kmeanstotalTime = duration<double>(t_kmeans_end - t_kmeans_start).count();
//
//    cout << "K-Means finished in " << iterations << " iterations, " << endl;
//    float lastMaxChange = (finalMaxChange.empty() ? 0.0f : finalMaxChange.back());
//    cout << "Last iteration max centroid change (max_change): " << lastMaxChange << endl;
//    cout << "Kmeans total time: " << kmeanstotalTime << " s " << endl;
//
//    // ======== ������������ؼ��޸���========
//    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
//    if (!driver) {
//        cerr << "GTiff driver not available!" << endl;
//        GDALClose(dataset);
//        return -1;
//    }
//
//    const char* outFile = "data/tm2002_kmeans_cpu.tif";
//    GDALDataset* outDs = driver->Create(outFile, width, height, 1, GDT_Byte, nullptr);
//    if (!outDs) {
//        cerr << "Failed to create output file!" << endl;
//        GDALClose(dataset);
//        return -1;
//    }
//
//    // ���Ƶ����任��������������任��
//    double geoTransform[6];
//    if (dataset->GetGeoTransform(geoTransform) == CE_None) {
//        outDs->SetGeoTransform(geoTransform);
//    }
//
//    // ����ͶӰ��Ϣ��WKT��ʽ��
//    const char* proj = dataset->GetProjectionRef();
//    if (proj && strlen(proj) > 0) {
//        outDs->SetProjection(proj);
//    }
//
//    // д������ǩ��0~K-1��
//    vector<unsigned char> outBuf(width * height);
//    for (long long i = 0; i < width * height; ++i) {
//        outBuf[i] = static_cast<unsigned char>(labels[i]);
//    }
//
//    GDALRasterBand* outBand = outDs->GetRasterBand(1);
//    outBand->RasterIO(GF_Write, 0, 0, width, height, outBuf.data(), width, height, GDT_Byte, 0, 0);
//    auto t_end = high_resolution_clock::now();
//    double totalTime = duration<double>(t_end - t_start).count();
//
//    cout << "Total time : " << totalTime << " s\n";
//
//    GDALClose(outDs);
//    GDALClose(dataset);
//
//    cout << "Output is saved as： " << outFile << endl;
//
//    return 0;
//}