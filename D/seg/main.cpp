#include "pcl/pcl_macros.h"
#include <iostream>
#include <random>
#include <set>
#include <vector>
#include <cmath>
#include <limits>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <ctime>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>

float angleBetweenNormals(const pcl::PointXYZRGBNormal& a, const pcl::PointXYZRGBNormal& b) {
    Eigen::Vector3f normalA(a.normal_x, a.normal_y, a.normal_z);
    Eigen::Vector3f normalB(b.normal_x, b.normal_y, b.normal_z);

    // 计算法向量的点积
    float dotProduct = normalA.dot(normalB);

    // 计算模长
    float magnitudeA = normalA.norm();
    float magnitudeB = normalB.norm();

    // 计算夹角（弧度）
    return std::acos(dotProduct / (magnitudeA * magnitudeB));
}
float computeCurvature(const pcl::PointXYZRGBNormal& point, const pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud, float radius) {
    // 使用KD树搜索邻域
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
    kdtree.setInputCloud(cloud.makeShared());

    std::vector<int> pointIndices;
    std::vector<float> pointSquaredDistances;

    if (kdtree.radiusSearch(point, radius, pointIndices, pointSquaredDistances) <= 0) {
        return 0.0f; // 如果没有找到邻域点，返回0
    }

    // 构建邻域点矩阵
    Eigen::MatrixXd pointsMatrix(pointIndices.size(), 3);
    for (size_t i = 0; i < pointIndices.size(); ++i) {
        const pcl::PointXYZRGBNormal& p = cloud.points[pointIndices[i]];
        pointsMatrix(i, 0) = p.x;
        pointsMatrix(i, 1) = p.y;
        pointsMatrix(i, 2) = p.z;
    }

    // 计算点云的协方差矩阵
    Eigen::Vector3d centroid = pointsMatrix.colwise().mean();
    Eigen::MatrixXd centered = pointsMatrix.rowwise() - centroid.transpose();
    Eigen::MatrixXd covariance = (centered.transpose() * centered) / double(pointsMatrix.rows() - 1);

    // 进行特征值分解，得到主曲率
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
    Eigen::Vector3d eigenValues = solver.eigenvalues();

    // 特征值从小到大排列
    std::sort(eigenValues.data(), eigenValues.data() + eigenValues.size());

    // 主曲率是两个最小的特征值（假设数据是曲面上的点云）
    float k1 = eigenValues[0];  // 最小的特征值
    float k2 = eigenValues[1];  // 第二小的特征值

    // 平均曲率 H = (k1 + k2) / 2
    float meanCurvature = (k1 + k2) / 2.0f;

    return meanCurvature;
}

float CCDistance(const pcl::PointXYZRGBNormal& a, const pcl::PointXYZRGBNormal& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) +
                     (a.y - b.y) * (a.y - b.y) +
                     (a.z - b.z) * (a.z - b.z));
}

float rgbDistance(const pcl::PointXYZRGBNormal& a, const pcl::PointXYZRGBNormal& b) {
    return std::sqrt((a.r - b.r) * (a.r - b.r) +
                     (a.g - b.g) * (a.g - b.g) +
                     (a.b - b.b) * (a.b - b.b));
}

float normalSimilarity(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud, int idx1, int idx2) {
    // 从点云中直接读取法向量
    Eigen::Vector3f normal1(cloud->points[idx1].normal_x, cloud->points[idx1].normal_y, cloud->points[idx1].normal_z);
    Eigen::Vector3f normal2(cloud->points[idx2].normal_x, cloud->points[idx2].normal_y, cloud->points[idx2].normal_z);

    // 计算法线之间的夹角余弦值
    float cosTheta = normal1.dot(normal2) / (normal1.norm() * normal2.norm());

    // 限制余弦值的范围，避免数值误差导致的越界
    cosTheta = std::min(1.0f, std::max(-1.0f, cosTheta));

    // 计算夹角（弧度）
    return std::acos(cosTheta);  // 返回弧度值，即法线之间的弧长
}

void regionGrowing(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud,
                   const std::vector<int>& seedIndices,
                   float distanceThreshold, float normalThreshold, float qlThreshold, float rgbThreshold,
                   const pcl::PointCloud<pcl::Normal>::Ptr& normals,
                   std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& outputClouds,
                   std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& outputClouds_w,
                   int minRegionSize, int maxAdditionalSeeds,float km,int k_1) {

    // 创建 KdTree 索引，优化 K 最近邻搜索
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
    kdtree.setInputCloud(cloud);

    // 初始化区域标记
    std::vector<int> regionLabels(cloud->size(), -1);  // 默认所有点未标记区域
    int currentRegionId = 0;  // 当前区域 ID

    // 初始种子点集合
    std::vector<int> seedPoints;
    if (seedIndices.empty()) {
        // 按顺序取前5个种子点
        for (int i = 0; i < 12&& i < cloud->size(); ++i) {
            seedPoints.push_back(i);
        }
    } else {
        // 使用提供的种子索引
        seedPoints = seedIndices;
    }

    // 清空并重新初始化输出区域点云
    outputClouds.clear();
    outputClouds_w.clear();
    outputClouds.resize(seedPoints.size());
    outputClouds_w.resize(seedPoints.size());    // 每个种子点对应一个输出区域

    // 对每个种子点进行独立区域生长
    for (int seedIdx : seedPoints) {
        if (regionLabels[seedIdx] != -1) continue;  // 如果该点已经被标记为区域，跳过

        // 为当前种子点分配一个新的区域ID
        regionLabels[seedIdx] = currentRegionId;
        std::set<int> toProcessLocal;
        toProcessLocal.insert(seedIdx);

        outputClouds[currentRegionId] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        outputClouds_w[currentRegionId] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        // 在当前种子点的领域内进行区域生长
        while (!toProcessLocal.empty()) {
            int currentIdx = *toProcessLocal.begin();
            toProcessLocal.erase(toProcessLocal.begin());

            outputClouds[currentRegionId]->points.push_back(cloud->points[currentIdx]);

            std::vector<int> pointIdxKSearch(k_1);  // 存储最近的 K 个邻居（K = 200）
            std::vector<float> pointKSearchSquaredDistance(30);  // 存储每个邻居的平方距离

            // 使用 KdTree 进行 KNN 查询，K = 10
            int numNeighbors = kdtree.nearestKSearch(cloud->points[currentIdx], 10, pointIdxKSearch, pointKSearchSquaredDistance);

            if (numNeighbors > 0) {
                for (size_t i = 0; i < pointIdxKSearch.size(); ++i) {
                    int neighborIdx = pointIdxKSearch[i];
                    if (neighborIdx == currentIdx || regionLabels[neighborIdx] != -1) continue;

                    float normalSim = normalSimilarity(cloud, currentIdx, neighborIdx);
                    float colorRgbDist = rgbDistance(cloud->points[currentIdx], cloud->points[neighborIdx]);
                    //float curvature = computeCurvature(cloud->points[currentIdx], *cloud, 2.0f);

                    // bool isSimilar =(normalSim <= normalThreshold && curvature < qlThreshold) ||
                    //                  (normalSim > normalThreshold && colorRgbDist < rgbThreshold);
                    bool cdSimilar =(normalSim <= normalThreshold ) ;
                    bool rgbSimilar =(colorRgbDist <= rgbThreshold) ;
                    int m=cloud->size();

                    float th1 = (normalThreshold * km) / m;
                    float th2 = (normalThreshold)/(k_1*m);
                    float rgb1 = (rgbThreshold * km) / m;
                    float rgb2 = (rgbThreshold)/(k_1*m);
                    //cout<<th2<<endl;
                    //cout<<th1<<endl;


                    if (cdSimilar) {
                        regionLabels[neighborIdx] = currentRegionId;
                        toProcessLocal.insert(neighborIdx);
                        normalThreshold=normalThreshold+ th1;
                        //rgbThreshold=rgbThreshold+rgb1;


                        // outFile_1 << normalSim << "," << curvature << std::endl;
                    }

                    // else if(cdSimilar!=1 && rgbsimlar){
                    //     normalThreshold=normalThreshold-0.00001;
                    //     regionLabels[neighborIdx] = currentRegionId;
                    //     toProcessLocal.insert(neighborIdx);
                    //     rgbThreshold=rgbThreshold+0.0001;
                    //  }
                    // else if(cdSimilar!=1 && rgbsimlar !=1)
                    // {
                    //     normalThreshold=normalThreshold-0.00001;
                    //     rgbThreshold=rgbThreshold-0.0001;
                    // }
                    else{
                        normalThreshold=normalThreshold-th2;
                           //cout<<normalThreshold<<endl;
                        //rgbThreshold=rgbThreshold+rgb2;
                    }
                }
            }
        }
        ++currentRegionId;
    }

    // 提取每个区域的点作为输出
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> regionClouds(currentRegionId);
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (regionLabels[i] != -1) {
            int regionId = regionLabels[i];
            if (!regionClouds[regionId]) {
                regionClouds[regionId] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            }
            regionClouds[regionId]->points.push_back(cloud->points[i]);
        }
    }

    outputClouds = regionClouds;

    // 修改未标记点的保存部分
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> regionClouds_w(currentRegionId + 1); // 预留一部分空间存放未标记点
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (regionLabels[i] == -1) {
            // 未被标记的点应存入 regionClouds_w 中
            if (!regionClouds_w[0]) {  // 第一个区域存放所有未标记的点
                regionClouds_w[0] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            }
            regionClouds_w[0]->points.push_back(cloud->points[i]);
        }
    }
    outputClouds_w = regionClouds_w;

}
// void savePointCloudToTXT(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud, const std::string& filename) {
//     // 打开输出文件
//     std::ofstream txtFile(filename);
//     if (!txtFile.is_open()) {
//         std::cerr << "Failed to open file: " << filename << std::endl;
//         return;
//     }

//     // 遍历每个点，并将其坐标、颜色、法线信息写入文件
//     for (const auto& point : cloud->points) {
//         txtFile << point.x << " "   // x坐标
//                 << point.y << " "   // y坐标
//                 << point.z << " "   // z坐标
//                 << static_cast<int>(point.r) << " "  // 红色分量
//                 << static_cast<int>(point.g) << " "  // 绿色分量
//                 << static_cast<int>(point.b) << " "  // 蓝色分量
//                 << point.normal_x << " "  // 法线x分量
//                 << point.normal_y << " "  // 法线y分量
//                 << point.normal_z << "\n";  // 法线z分量
//     }

//     // 关闭文件
//     txtFile.close();
//     std::cout << "Point cloud saved to " << filename << std::endl;
// }




// 区域生长函数
// 主程序
int main() {
    // 示例数据：创建一些点云数据
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal> ("D:\\Desktop\\LS\\4.22\\mox3\\chusd.pcd", *cloud) == -1) { //* load the file
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }
    // std::ofstream txt_File("D:\\Desktop\\LS\\11.13\\lbpcd.txt");
    // savePointCloudToTXT(cloud, "D:\\Desktop\\LS\\11.13\\lbpcd.txt");
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normals->points.resize(cloud->points.size());
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        normals->points[i].normal_x = cloud->points[i].normal_x;
        normals->points[i].normal_y = cloud->points[i].normal_y;
        normals->points[i].normal_z = cloud->points[i].normal_z;
    }

    // 3. 定义区域生长算法的参数
    float distanceThreshold = 1.5;  // 欧几里得距离阈值
    int k_1=30;
    float km=0.5;
    float normalThreshold = ((2*M_PI)/180);     // 法线相似度阈值
    float qlThreshold = 2.0f;         // 曲率阈值（用于分割）
    float rgbThreshold = 3.0f;       // RGB颜色阈值
    std::vector<int> seedIndices = {}; // 空的种子点集合
    int minRegionSize = 200;  // 最小区域大小
    int maxAdditionalSeeds = 0;  // 最大额外种子点数量


    // 创建输出点云
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> outputClouds;
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> outputClouds_w;

    cout<<1<<endl;
    // 调用区域生长函数
    regionGrowing(cloud, seedIndices, distanceThreshold, normalThreshold, qlThreshold, rgbThreshold, normals, outputClouds,outputClouds_w,
                  minRegionSize, maxAdditionalSeeds,km,k_1);

    cout<<2<<endl;

    for (size_t i = 0; i < outputClouds.size(); ++i) {
        try {
            if (outputClouds[i] == nullptr) {
                cout << "Cloud " << i << " is null!" << endl;
                continue;
            }

            size_t point_count = outputClouds[i]->points.size();
            if (point_count == 0) {
                cout << "Cloud " << i << " has no points!" << endl;
                continue;
            }

            // 设置 width 和 height
            outputClouds[i]->width = point_count;  // 设置宽度为点的数量
            outputClouds[i]->height = 1;  // 设置为无序点云

            std::stringstream ss;
            ss << "D:\\Desktop\\LS\\4.22\\mox3\\mox3_fg_1.0\\fgbf" << i << ".pcd";  // 定义保存路径

            // 保存点云文件
            pcl::io::savePCDFile(ss.str(), *outputClouds[i]);
            cout << "Saved cloud " << i << " to " << ss.str() << outputClouds[i]->size() <<endl;
        }
        catch (const std::exception& e) {
            cout << "Error saving cloud " << i << ": " << e.what() << endl;
        }
    }
    for (size_t i = 0; i < outputClouds_w.size(); ++i) {
        try {
            if (outputClouds_w[i] == nullptr) {
                cout << "Cloud " << i << " is null!" << endl;
                continue;
            }

            size_t point_count = outputClouds_w[i]->points.size();
            if (point_count == 0) {
                cout << "Cloud " << i << " has no points!" << endl;
                continue;
            }

            // 设置 width 和 height
            outputClouds_w[i]->width = point_count;  // 设置宽度为点的数量
            outputClouds_w[i]->height = 1;  // 设置为无序点云

            std::stringstream ss;
            ss << "D:\\Desktop\\LS\\4.22\\mox3\\mox3_fg_1.0\\wfgbf" << i << ".pcd";  // 定义保存路径

            // 保存点云文件
            pcl::io::savePCDFile(ss.str(), *outputClouds_w[i]);
            cout << "Saved cloud " << i << " to " << ss.str() << outputClouds_w[i]->size() <<endl;
        }
        catch (const std::exception& e) {
            cout << "Error saving cloud " << i << ": " << e.what() << endl;
        }
    }
    return 0;
}

