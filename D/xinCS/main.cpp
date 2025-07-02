#include <iostream>
#include <set>
#include <vector>
#include <cmath>
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
#include <jcgs.h>

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


float normalSimilarity(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud, int idx1, int idx2) {
    // 从点云中直接读取法向量
    Eigen::Vector3f normal1(cloud->points[idx1].normal_x, cloud->points[idx1].normal_y, cloud->points[idx1].normal_z);
    Eigen::Vector3f normal2(cloud->points[idx2].normal_x, cloud->points[idx2].normal_y, cloud->points[idx2].normal_z);

    // 计算法线之间的夹角余弦值
    float cosTheta = normal1.dot(normal2) / (normal1.norm() * normal2.norm());

    // 计算法线差异
    return 1.0f - cosTheta;
}


float calculateRegionSimilarity(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud,
                                const std::vector<int>& regionLabels, int region1, int region2) {
    // 提取两个区域的点
    std::vector<pcl::PointXYZRGBNormal> region1Points, region2Points;
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (regionLabels[i] == region1) {
            region1Points.push_back(cloud->points[i]);
        } else if (regionLabels[i] == region2) {
            region2Points.push_back(cloud->points[i]);
        }
    }

    // 计算法线差异和颜色差异的加权平均值
    float normalDifference = 0.0f;
    float colorDifference = 0.0f;

    for (const auto& point1 : region1Points) {
        for (const auto& point2 : region2Points) {
            // 计算法线差异：点积
            float normalSim = point1.normal_x * point2.normal_x +
                              point1.normal_y * point2.normal_y +
                              point1.normal_z * point2.normal_z;
            normalDifference += (1.0f - normalSim);  // 法线差异为 1 - 点积

            // 计算颜色差异：欧几里得距离
            float colorSim = std::sqrt(std::pow(point1.r - point2.r, 2) +
                                       std::pow(point1.g - point2.g, 2) +
                                       std::pow(point1.b - point2.b, 2));
            colorDifference += colorSim;
        }
    }
    // 使用法线差异和颜色差异的加权和来表示区域之间的相似度
    // 你可以根据实际情况调整权重
    float totalSimilarity = 1.0f / (normalDifference + 0.5f * colorDifference + 0.1f);
    return totalSimilarity;
}

bool areRegionsAdjacentOrOverlapping(const std::vector<int>& regionLabels,
                                     const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud,
                                     int region1, int region2) {
    std::set<int> region1Points, region2Points;

    // 获取区域1和区域2的点集合
    for (size_t i = 0; i < regionLabels.size(); ++i) {
        if (regionLabels[i] == region1) {
            region1Points.insert(i);
        } else if (regionLabels[i] == region2) {
            region2Points.insert(i);
        }
    }

    // 对于每个区域1的点，检查其邻域是否包含区域2的点
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
    kdtree.setInputCloud(cloud);

    for (int idx1 : region1Points) {
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        int numNeighbors = kdtree.radiusSearch(cloud->points[idx1], 0.1, pointIdxRadiusSearch, pointRadiusSquaredDistance); // 设置合适的半径

        for (int idx2 : region2Points) {
            if (std::find(pointIdxRadiusSearch.begin(), pointIdxRadiusSearch.end(), idx2) != pointIdxRadiusSearch.end()) {
                return true; // 如果区域1中的点与区域2中的点有共享的邻域，说明区域相邻
            }
        }
    }

    return false; // 没有共享邻域，返回false
}

void mergeSmallRegions(std::vector<int>& regionLabels,
                       const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud,
                       int numRegions, int minRegionSize) {
    // 统计每个区域的点数量
    std::vector<int> regionSizes(numRegions, 0);
    for (int label : regionLabels) {
        if (label != -1) {
            regionSizes[label]++;
        }
    }

    // 合并小区域
    for (int i = 0; i < numRegions; ++i) {
        if (regionSizes[i] < minRegionSize) {
            // 找到最相似的区域，并将小区域的点合并到相似的区域
            int bestMatchRegion = -1;
            float bestSimilarity = -1.0;

            // 遍历其他区域，查找最相似的区域
            for (int j = 0; j < numRegions; ++j) {
                if (i != j && regionSizes[j] > minRegionSize) {
                    // 检查区域i和区域j是否相邻或重叠
                    if (areRegionsAdjacentOrOverlapping(regionLabels, cloud, i, j)) {
                        float similarity = calculateRegionSimilarity(cloud, regionLabels, i, j);
                        if (similarity > bestSimilarity) {
                            bestSimilarity = similarity;
                            bestMatchRegion = j;
                        }
                    }
                }
            }

            // 将小区域的点标记为目标区域
            if (bestMatchRegion != -1) {
                for (int k = 0; k < regionLabels.size(); ++k) {
                    if (regionLabels[k] == i) {
                        regionLabels[k] = bestMatchRegion;
                    }
                }
            }
        }
    }
}

void regionGrowing(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud,
                   const std::vector<int>& seedIndices,
                   float distanceThreshold, float normalThreshold, float qlThreshold, float rgbThreshold,
                   const pcl::PointCloud<pcl::Normal>::Ptr& normals,
                   std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& outputClouds,
                   int minRegionSize, int maxAdditionalSeeds) {

    // 创建 KdTree 索引，优化 K 最近邻搜索
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
    kdtree.setInputCloud(cloud);

    std::ofstream outFile("D:\\Desktop\\LS\\11.13\\qulv_0.csv");
    std::ofstream txtFile("D:\\Desktop\\LS\\11.13\\pcd.txt");

    std::ofstream outFile_1("D:\\Desktop\\LS\\11.13\\qulv_1.csv");

    // 初始化区域标记
    std::vector<int> regionLabels(cloud->size(), -1);  // 默认所有点未标记区域
    int currentRegionId = 0;  // 当前区域 ID

    // 初始种子点集合
    std::vector<int> seedPoints;
    if (seedIndices.empty()) {
        srand(time(0));
        while (seedPoints.size() < 5) {
            int randomIdx = rand() % cloud->size();
            if (std::find(seedPoints.begin(), seedPoints.end(), randomIdx) == seedPoints.end()) {
                seedPoints.push_back(randomIdx);
            }
        }
    } else {
        seedPoints = seedIndices;
    }

    // 清空并重新初始化输出区域点云
    outputClouds.clear();
    outputClouds.resize(seedPoints.size());  // 每个种子点对应一个输出区域

    // 对每个种子点进行独立区域生长
    for (int seedIdx : seedPoints) {
        if (regionLabels[seedIdx] != -1) continue;  // 如果该点已经被标记为区域，跳过

        // 为当前种子点分配一个新的区域ID
        regionLabels[seedIdx] = currentRegionId;
        std::set<int> toProcessLocal;
        toProcessLocal.insert(seedIdx);

        outputClouds[currentRegionId] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        // 在当前种子点的领域内进行区域生长
        while (!toProcessLocal.empty()) {
            int currentIdx = *toProcessLocal.begin();
            toProcessLocal.erase(toProcessLocal.begin());

            outputClouds[currentRegionId]->points.push_back(cloud->points[currentIdx]);

            std::vector<int> pointIdxKSearch(10);  // 存储最近的 K 个邻居（K = 10）
            std::vector<float> pointKSearchSquaredDistance(10);  // 存储每个邻居的平方距离

            // 使用 KdTree 进行 KNN 查询，K = 10
            int numNeighbors = kdtree.nearestKSearch(cloud->points[currentIdx], 10, pointIdxKSearch, pointKSearchSquaredDistance);

            if (numNeighbors > 0) {
                for (size_t i = 0; i < pointIdxKSearch.size(); ++i) {
                    int neighborIdx = pointIdxKSearch[i];
                    if (neighborIdx == currentIdx || regionLabels[neighborIdx] != -1) continue;

                    float normalSim = normalSimilarity(cloud, currentIdx, neighborIdx);
                    float colorRgbDist = rgbDistance(cloud->points[currentIdx], cloud->points[neighborIdx]);
                    float curvature = computeCurvature(cloud->points[currentIdx], *cloud, 2.0f);

                    //bool isSimilar =(normalSim <= normalThreshold && curvature < qlThreshold) ||
                                     //(normalSim > normalThreshold && colorRgbDist < rgbThreshold);

                    bool isSimilar = (normalSim <= normalThreshold );
                    //bool isSimilar = (colorRgbDist<=rgbThreshold);
                    // outFile << normalSim << "," << curvature << std::endl;
                    // if (normalSim > 0.02) {
                    //     // 输出种子点的信息到 txtFile
                    //     const auto& seedPoint = cloud->points[seedIdx];
                    //     const auto& neighborPoint = cloud->points[neighborIdx];

                    //     // 记录种子点的信息
                    //     txtFile  << seedPoint.x << " " << seedPoint.y << " " << seedPoint.z
                    //             << " " << int(seedPoint.r) << " " << int(seedPoint.g) << " " << int(seedPoint.b)
                    //             << " " << seedPoint.normal_x << " " << seedPoint.normal_y << " " << seedPoint.normal_z
                    //             << std::endl;

                    //     // 记录邻域点的信息
                    //     txtFile<< neighborPoint.x << " " << neighborPoint.y << " " << neighborPoint.z << " "
                    //             << int(neighborPoint.r) << " " << int(neighborPoint.g) << " " << int(neighborPoint.b) << " "
                    //             << neighborPoint.normal_x << " " << neighborPoint.normal_y << " " << neighborPoint.normal_z
                    //             << std::endl;
                    // }

                    if (isSimilar) {
                        regionLabels[neighborIdx] = currentRegionId;
                        toProcessLocal.insert(neighborIdx);
                        // outFile_1 << normalSim << "," << curvature << std::endl;
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
}
void savePointCloudToTXT(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud, const std::string& filename) {
    // 打开输出文件
    std::ofstream txtFile(filename);
    if (!txtFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // 遍历每个点，并将其坐标、颜色、法线信息写入文件
    for (const auto& point : cloud->points) {
        txtFile << point.x << " "   // x坐标
                << point.y << " "   // y坐标
                << point.z << " "   // z坐标
                << static_cast<int>(point.r) << " "  // 红色分量
                << static_cast<int>(point.g) << " "  // 绿色分量
                << static_cast<int>(point.b) << " "  // 蓝色分量
                << point.normal_x << " "  // 法线x分量
                << point.normal_y << " "  // 法线y分量
                << point.normal_z << "\n";  // 法线z分量
    }

    // 关闭文件
    txtFile.close();
    std::cout << "Point cloud saved to " << filename << std::endl;
}




// 区域生长函数
// 主程序
int main() {
    // 示例数据：创建一些点云数据
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal> ("D:\\Desktop\\LS\\qtxin_lb.pcd", *cloud) == -1) { //* load the file
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
    float distanceThreshold = 1.2;  // 欧几里得距离阈值
    float normalThreshold = 0.0006f;     // 法线相似度阈值
    float qlThreshold = 1.0f;         // 曲率阈值（用于分割）
    float rgbThreshold = 10.0f;       // RGB颜色阈值
    std::vector<int> seedIndices = {}; // 空的种子点集合
    int minRegionSize = 200;  // 最小区域大小
    int maxAdditionalSeeds = 0;  // 最大额外种子点数量


    // 创建输出点云
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> outputClouds;

    cout<<1<<endl;
    // 调用区域生长函数
    regionGrowing(cloud, seedIndices, distanceThreshold, normalThreshold, qlThreshold, rgbThreshold, normals, outputClouds, minRegionSize, maxAdditionalSeeds);

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
            ss << "D:\\Desktop\\LS\\tqxincs_" << i << ".pcd";  // 定义保存路径

            // 保存点云文件
            pcl::io::savePCDFile(ss.str(), *outputClouds[i]);
            cout << "Saved cloud " << i << " to " << ss.str() << outputClouds[i]->size() <<endl;
        }
        catch (const std::exception& e) {
            cout << "Error saving cloud " << i << ": " << e.what() << endl;
        }
    }
    return 0;
}

