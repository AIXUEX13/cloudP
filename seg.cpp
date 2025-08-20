#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/principal_curvatures.h>
#include <vector>
#include <algorithm>
#include <pcl/io/pcd_io.h>
#include <fstream>

#include "pcl/pcl_macros.h"
#include <iostream>
#include <random>
#include <set>
#include <cmath>
#include <limits>
#include <pcl/octree/octree_search.h>
#include <ctime>

#include <pcl/kdtree/kdtree.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>
#include <thread>
#include <chrono>

#include <pcl/segmentation/extract_clusters.h>

    // ---------------------- helpers ----------------------
    constexpr double PI_CONST = 3.14159265358979323846;

struct PointWithCurvature {
    pcl::PointXYZRGBNormal point;
    float curvature;
    bool operator<(const PointWithCurvature& other) const {
        return curvature < other.curvature;
    }
};

// 计算主曲率并按曲率升序重排 cloudXYZRGBNormal，同时把 XYZ 输出到 cloudXYZSorted
void computeCurvaturesAndSort(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudXYZRGBNormal,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudXYZSorted) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
    cloudXYZ->reserve(cloudXYZRGBNormal->points.size());
    for (size_t i = 0; i < cloudXYZRGBNormal->points.size(); ++i) {
        cloudXYZ->push_back(pcl::PointXYZ(cloudXYZRGBNormal->points[i].x,
                                          cloudXYZRGBNormal->points[i].y,
                                          cloudXYZRGBNormal->points[i].z));
    }

    // 法线已存在于 cloudXYZRGBNormal
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    normals->reserve(cloudXYZRGBNormal->points.size());
    for (size_t i = 0; i < cloudXYZRGBNormal->points.size(); ++i) {
        pcl::Normal n;
        n.normal_x = cloudXYZRGBNormal->points[i].normal_x;
        n.normal_y = cloudXYZRGBNormal->points[i].normal_y;
        n.normal_z = cloudXYZRGBNormal->points[i].normal_z;
        normals->push_back(n);
    }

    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> curvatureEstimation;
    curvatureEstimation.setInputCloud(cloudXYZ);
    curvatureEstimation.setInputNormals(normals);
    curvatureEstimation.setRadiusSearch(1.2f);

    pcl::PointCloud<pcl::PrincipalCurvatures> curvatures;
    curvatureEstimation.compute(curvatures);

    std::vector<PointWithCurvature> pointsWithCurvature;
    pointsWithCurvature.reserve(cloudXYZ->size());
    for (size_t i = 0; i < cloudXYZ->size(); ++i) {
        PointWithCurvature pwc;
        pwc.point = cloudXYZRGBNormal->points[i];
        pwc.curvature = std::min(std::fabs(curvatures.points[i].pc1),
                                 std::fabs(curvatures.points[i].pc2));
        pointsWithCurvature.push_back(pwc);
    }

    std::sort(pointsWithCurvature.begin(), pointsWithCurvature.end());

    // 用排序结果重建
    cloudXYZRGBNormal->clear();
    cloudXYZSorted->clear();
    cloudXYZRGBNormal->reserve(pointsWithCurvature.size());
    cloudXYZSorted->reserve(pointsWithCurvature.size());
    for (const auto& pwc : pointsWithCurvature) {
        cloudXYZRGBNormal->push_back(pwc.point);
        cloudXYZSorted->push_back(pcl::PointXYZ(pwc.point.x, pwc.point.y, pwc.point.z));
    }
}

float angleBetweenNormals(const pcl::PointXYZRGBNormal& a, const pcl::PointXYZRGBNormal& b) {
    Eigen::Vector3f normalA(a.normal_x, a.normal_y, a.normal_z);
    Eigen::Vector3f normalB(b.normal_x, b.normal_y, b.normal_z);
    float denom = normalA.norm() * normalB.norm();
    if (denom == 0.0f) return 0.0f;
    float cosv = normalA.dot(normalB) / denom;
    // clamp 避免 NaN
    cosv = std::max(-1.0f, std::min(1.0f, cosv));
    return std::acos(cosv);
}

float computeCurvature(const pcl::PointXYZRGBNormal& point, const pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud, float radius) {
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
    kdtree.setInputCloud(cloud.makeShared());

    std::vector<int> pointIndices;
    std::vector<float> pointSquaredDistances;

    if (kdtree.radiusSearch(point, radius, pointIndices, pointSquaredDistances) <= 0) {
        return 0.0f;
    }

    Eigen::MatrixXd pointsMatrix(pointIndices.size(), 3);
    for (size_t i = 0; i < pointIndices.size(); ++i) {
        const auto& p = cloud.points[pointIndices[i]];
        pointsMatrix(i, 0) = p.x;
        pointsMatrix(i, 1) = p.y;
        pointsMatrix(i, 2) = p.z;
    }

    Eigen::Vector3d centroid = pointsMatrix.colwise().mean();
    Eigen::MatrixXd centered = pointsMatrix.rowwise() - centroid.transpose();
    Eigen::MatrixXd covariance = (centered.transpose() * centered) / double(pointsMatrix.rows() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
    Eigen::Vector3d eigenValues = solver.eigenvalues();

    // 升序
    std::sort(eigenValues.data(), eigenValues.data() + eigenValues.size());
    float k1 = static_cast<float>(eigenValues[0]);
    float k2 = static_cast<float>(eigenValues[1]);
    return (k1 + k2) * 0.5f;
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
    Eigen::Vector3f normal1(cloud->points[idx1].normal_x, cloud->points[idx1].normal_y, cloud->points[idx1].normal_z);
    Eigen::Vector3f normal2(cloud->points[idx2].normal_x, cloud->points[idx2].normal_y, cloud->points[idx2].normal_z);

    float denom = normal1.norm() * normal2.norm();
    if (denom == 0.0f) return 0.0f;
    float cosTheta = normal1.dot(normal2) / denom;
    cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta));
    return std::acos(cosTheta);
}

void regionGrowing_RGB(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud,
                       const std::vector<int>& seedIndices,
                       float /*distanceThreshold*/, float normalThreshold, float /*qlThreshold*/, float rgbThreshold,
                       const pcl::PointCloud<pcl::Normal>::Ptr& /*normals*/,
                       std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& outputClouds,
                       std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& outputClouds_w,
                       int /*minRegionSize*/, int /*maxAdditionalSeeds*/, float km, int k_1) {

    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<int> regionLabels(cloud->size(), -1);
    int currentRegionId = 0;

    std::vector<int> seedPoints;
    if (seedIndices.empty()) {
        for (int i = 0; i < 1 && i < static_cast<int>(cloud->size()); ++i) {
            seedPoints.push_back(i);
        }
    } else {
        seedPoints = seedIndices;
    }

    outputClouds.clear();
    outputClouds_w.clear();
    outputClouds.resize(seedPoints.size());
    outputClouds_w.resize(seedPoints.size());

    for (int seedIdx : seedPoints) {
        if (regionLabels[seedIdx] != -1) continue;

        regionLabels[seedIdx] = currentRegionId;
        std::set<int> toProcessLocal;
        toProcessLocal.insert(seedIdx);

        outputClouds[currentRegionId] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        outputClouds_w[currentRegionId] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        while (!toProcessLocal.empty()) {
            int currentIdx = *toProcessLocal.begin();
            toProcessLocal.erase(toProcessLocal.begin());

            outputClouds[currentRegionId]->push_back(cloud->points[currentIdx]);

            int K = std::max(1, k_1);
            std::vector<int> pointIdxKSearch(K);
            std::vector<float> pointKSearchSquaredDistance(K);

            int numNeighbors = kdtree.nearestKSearch(cloud->points[currentIdx], K, pointIdxKSearch, pointKSearchSquaredDistance);

            if (numNeighbors > 0) {
                for (int i = 0; i < numNeighbors; ++i) {
                    int neighborIdx = pointIdxKSearch[i];
                    if (neighborIdx == currentIdx || regionLabels[neighborIdx] != -1) continue;

                    float normalSim = normalSimilarity(cloud, currentIdx, neighborIdx);
                    float colorRgbDist = rgbDistance(cloud->points[currentIdx], cloud->points[neighborIdx]);

                    bool cdSimilar = (normalSim <= normalThreshold);
                    bool rgbSimilar = (colorRgbDist <= rgbThreshold);
                    int m = static_cast<int>(cloud->size());

                    float th1 = (normalThreshold * km) / m;
                    float th2 = (normalThreshold) / (k_1 * m);
                    float rgb1 = (rgbThreshold * km) / m;
                    float rgb2 = (rgbThreshold) / (k_1 * m);

                    if (rgbSimilar) {
                        regionLabels[neighborIdx] = currentRegionId;
                        toProcessLocal.insert(neighborIdx);
                        (void)cdSimilar; // 未使用但保留变量
                        rgbThreshold = rgbThreshold + rgb1;
                    } else {
                        rgbThreshold = rgbThreshold + rgb2;
                    }
                }
            }
        }
        ++currentRegionId;
    }

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> regionClouds(currentRegionId);
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (regionLabels[i] != -1) {
            int regionId = regionLabels[i];
            if (!regionClouds[regionId]) {
                regionClouds[regionId] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            }
            regionClouds[regionId]->push_back(cloud->points[i]);
        }
    }
    outputClouds = regionClouds;

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> regionClouds_w(currentRegionId + 1);
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (regionLabels[i] == -1) {
            if (!regionClouds_w[0]) {
                regionClouds_w[0] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            }
            regionClouds_w[0]->push_back(cloud->points[i]);
        }
    }
    outputClouds_w = regionClouds_w;
}

void findCommonPoints(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud,
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud_A,
    std::vector<int>& seedIndices) {
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        for (size_t j = 0; j < cloud_A->points.size(); ++j) {
            if (cloud->points[i].x == cloud_A->points[j].x &&
                cloud->points[i].y == cloud_A->points[j].y &&
                cloud->points[i].z == cloud_A->points[j].z) {
                seedIndices.push_back(static_cast<int>(i));
                break;
            }
        }
    }
}

void regionGrowing(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud,
                   const std::vector<int>& seedIndices,
                   float /*distanceThreshold*/, float normalThreshold, float /*qlThreshold*/, float /*rgbThreshold*/,
                   const pcl::PointCloud<pcl::Normal>::Ptr& /*normals*/,
                   std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& outputClouds,
                   std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& outputClouds_w,
                   int /*minRegionSize*/, int /*maxAdditionalSeeds*/, float km, int k_1) {

    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<int> regionLabels(cloud->size(), -1);
    int currentRegionId = 0;

    std::vector<int> seedPoints;
    if (seedIndices.empty()) {
        for (int i = 130; i < 150 && i < static_cast<int>(cloud->size()); ++i) {
            seedPoints.push_back(i);
        }
    } else {
        seedPoints = seedIndices;
    }

    outputClouds.clear();
    outputClouds_w.clear();
    outputClouds.resize(seedPoints.size());
    outputClouds_w.resize(seedPoints.size());

    for (int seedIdx : seedPoints) {
        if (regionLabels[seedIdx] != -1) continue;

        regionLabels[seedIdx] = currentRegionId;
        std::set<int> toProcessLocal;
        toProcessLocal.insert(seedIdx);

        outputClouds[currentRegionId] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        outputClouds_w[currentRegionId] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        while (!toProcessLocal.empty()) {
            int currentIdx = *toProcessLocal.begin();
            toProcessLocal.erase(toProcessLocal.begin());

            outputClouds[currentRegionId]->push_back(cloud->points[currentIdx]);

            int K = std::max(1, k_1);
            std::vector<int> pointIdxKSearch(K);
            std::vector<float> pointKSearchSquaredDistance(K);

            int numNeighbors = kdtree.nearestKSearch(cloud->points[currentIdx], K, pointIdxKSearch, pointKSearchSquaredDistance);

            if (numNeighbors > 0) {
                for (int i = 0; i < numNeighbors; ++i) {
                    int neighborIdx = pointIdxKSearch[i];
                    if (neighborIdx == currentIdx || regionLabels[neighborIdx] != -1) continue;

                    float normalSim = normalSimilarity(cloud, currentIdx, neighborIdx);
                    bool cdSimilar = (normalSim <= normalThreshold);
                    int m = static_cast<int>(cloud->size());

                    float th1 = (normalThreshold * km) / m;
                    float th2 = (normalThreshold) / (k_1 * m);

                    if (cdSimilar) {
                        regionLabels[neighborIdx] = currentRegionId;
                        toProcessLocal.insert(neighborIdx);
                        normalThreshold = normalThreshold + th1;
                    } else {
                        normalThreshold = normalThreshold - th2;
                    }
                }
            }
        }
        ++currentRegionId;
    }

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> regionClouds(currentRegionId);
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (regionLabels[i] != -1) {
            int regionId = regionLabels[i];
            if (!regionClouds[regionId]) {
                regionClouds[regionId] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            }
            regionClouds[regionId]->push_back(cloud->points[i]);
        }
    }
    outputClouds = regionClouds;

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> regionClouds_w(currentRegionId + 1);
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (regionLabels[i] == -1) {
            if (!regionClouds_w[0]) {
                regionClouds_w[0] = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            }
            regionClouds_w[0]->push_back(cloud->points[i]);
        }
    }
    outputClouds_w = regionClouds_w;
}

// 边界提取
void PointCloudBoundary2(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud , pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_boundary) {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setSearchMethod(tree);
    normalEstimation.setRadiusSearch(1);
    normalEstimation.compute(*normals);

    pcl::PointCloud<pcl::Boundary>::Ptr boundaries(new pcl::PointCloud<pcl::Boundary>);
    boundaries->resize(cloud->size());
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimation;
    boundary_estimation.setInputCloud(cloud);
    boundary_estimation.setInputNormals(normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_ptr(new pcl::search::KdTree<pcl::PointXYZ>);
    boundary_estimation.setSearchMethod(kdtree_ptr);
    boundary_estimation.setKSearch(30);
    boundary_estimation.setAngleThreshold(PI_CONST * 0.5);
    boundary_estimation.compute(*boundaries);

    for (size_t i = 0; i < cloud->size(); i++) {
        if (boundaries->points[i].boundary_point != 0) {
            pcl::PointXYZRGB p;
            p.x = cloud->points[i].x;
            p.y = cloud->points[i].y;
            p.z = cloud->points[i].z;
            p.r = 255; p.g = 0; p.b = 0;
            cloud_boundary->push_back(p);
        }
    }
}

// 配对
void peidui(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud_A,
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_B,
            const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud_C) {
    for (const auto& point_a : cloud_A->points) {
        for (const auto& point_b : cloud_B->points) {
            if (point_a.x == point_b.x && point_a.y == point_b.y && point_a.z == point_b.z) {
                bool point_exists = false;
                for (const auto& existing_point : cloud_C->points) {
                    if (existing_point.x == point_a.x && existing_point.y == point_a.y && existing_point.z == point_a.z &&
                        existing_point.r == point_a.r && existing_point.g == point_a.g && existing_point.b == point_a.b) {
                        point_exists = true;
                        break;
                    }
                }
                if (!point_exists) {
                    pcl::PointXYZRGBNormal point_c;
                    point_c.x = point_a.x; point_c.y = point_a.y; point_c.z = point_a.z;
                    point_c.r = point_a.r; point_c.g = point_a.g; point_c.b = point_a.b;
                    point_c.normal_x = point_a.normal_x; point_c.normal_y = point_a.normal_y; point_c.normal_z = point_a.normal_z;
                    cloud_C->points.push_back(point_c);
                }
            }
        }
    }
    cloud_C->width = static_cast<uint32_t>(cloud_C->points.size());
    cloud_C->height = 1;
}

// 聚类筛除小簇
void liqun(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud_A,
           const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud_C) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->resize(cloud_A->points.size());
    for (size_t i = 0; i < cloud_A->points.size(); ++i) {
        cloud->points[i].x = cloud_A->points[i].x;
        cloud->points[i].y = cloud_A->points[i].y;
        cloud->points[i].z = cloud_A->points[i].z;
        cloud->points[i].r = cloud_A->points[i].r;
        cloud->points[i].g = cloud_A->points[i].g;
        cloud->points[i].b = cloud_A->points[i].b;
    }

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    std::vector<pcl::PointIndices> cluster_indices;
    ec.setClusterTolerance(0.3);
    ec.setMinClusterSize(2000);
    ec.setMaxClusterSize(10000000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    std::cout << "Number of clusters: " << cluster_indices.size() << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_final(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (const auto& indices : cluster_indices) {
        if (indices.indices.size() >= 2000) {
            for (const auto& index : indices.indices) {
                cloud_filtered_final->points.push_back(cloud->points[index]);
            }
        }
    }
    cloud_filtered_final->width = static_cast<uint32_t>(cloud_filtered_final->points.size());
    cloud_filtered_final->height = 1;

    for (const auto& point_b : cloud_filtered_final->points) {
        for (const auto& point_a : cloud_A->points) {
            if (point_a.x == point_b.x && point_a.y == point_b.y && point_a.z == point_b.z &&
                point_a.r == point_b.r && point_a.g == point_b.g && point_a.b == point_b.b) {
                bool point_exists = false;
                for (const auto& existing_point : cloud_C->points) {
                    if (existing_point.x == point_a.x && existing_point.y == point_a.y && existing_point.z == point_a.z &&
                        existing_point.r == point_a.r && existing_point.g == point_a.g && existing_point.b == point_a.b) {
                        point_exists = true; break;
                    }
                }
                if (!point_exists) {
                    pcl::PointXYZRGBNormal point_c;
                    point_c.x = point_a.x; point_c.y = point_a.y; point_c.z = point_a.z;
                    point_c.r = point_a.r; point_c.g = point_a.g; point_c.b = point_a.b;
                    point_c.normal_x = point_a.normal_x; point_c.normal_y = point_a.normal_y; point_c.normal_z = point_a.normal_z;
                    cloud_C->points.push_back(point_c);
                }
            }
        }
    }
    cloud_C->width = static_cast<uint32_t>(cloud_C->points.size());
    cloud_C->height = 1;
}

// 颜色相似度（CVS）——通过引用返回
void CVS(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud_A,
         const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud,
         const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud_Z,
         float& cvs) {
    float num = 0.0f;
    for (const auto& point_a : cloud_A->points) {
        for (const auto& point_b : cloud->points) {
            float ra = point_a.r, ga = point_a.g, ba = point_a.b;
            float rb = point_b.r, gb = point_b.g, bb = point_b.b;
            float co = std::sqrt((ra - rb) * (ra - rb) + (ga - gb) * (ga - gb) + (ba - bb) * (ba - bb));
            if (co <= 1.5f) {
                num++;
                break;
            }
        }
    }
    if (!cloud_Z->empty()) {
        cvs = num / static_cast<float>(cloud_Z->size());
    } else {
        cvs = 0.0f;
    }
    std::cout << "cvs: " << (cvs * 100.0f) << "%\n";
    std::cout << "num: " << num << std::endl;
}

int main() {
    // ---- 读取点云 ----
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\4.22\\mox1\\chushidianpcd.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file chusd.pcd\\n");
        return -1;
    }
    pcl::io::savePCDFileASCII("D:\\Desktop\\LS\\cloud_cs1.pcd", *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sor(new pcl::PointCloud<pcl::PointXYZ>());
    computeCurvaturesAndSort(cloud, cloud_sor);

    pcl::io::savePCDFileASCII("D:\\Desktop\\LS\\cloud_cs2.pcd", *cloud);

    // 拷贝法线
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normals->resize(cloud->points.size());
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        (*normals)[i].normal_x = cloud->points[i].normal_x;
        (*normals)[i].normal_y = cloud->points[i].normal_y;
        (*normals)[i].normal_z = cloud->points[i].normal_z;
    }

    // 区域生长参数
    float distanceThreshold = 1.0f;
    int k_1 = 50;
    float km = 0.1f;
    float normalThreshold = ((1.2 * static_cast<float>(PI_CONST)) / 180.0f);
    float qlThreshold = 2.0f;
    float rgbThreshold = 1.0f;
    std::vector<int> seedIndices;
    int minRegionSize = 200;
    int maxAdditionalSeeds = 0;

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> outputClouds;
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> outputClouds_w;

    regionGrowing(cloud, seedIndices, distanceThreshold, normalThreshold, qlThreshold, rgbThreshold,
                  normals, outputClouds, outputClouds_w, minRegionSize, maxAdditionalSeeds, km, k_1);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_new(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    if (!outputClouds_w.empty() && outputClouds_w[0]) {
        liqun(outputClouds_w[0], cloud_new);
    }

    // --- 生成边界并配对 ---
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_b_a(new pcl::PointCloud<pcl::PointXYZ>());
    cloud_b_a->resize(cloud_new->size());
    for (size_t i = 0; i < cloud_new->size(); i++) {
        (*cloud_b_a)[i].x = cloud_new->points[i].x;
        (*cloud_b_a)[i].y = cloud_new->points[i].y;
        (*cloud_b_a)[i].z = cloud_new->points[i].z;
    }
    cloud_b_a->width = static_cast<uint32_t>(cloud_b_a->points.size());
    cloud_b_a->height = 1;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_b_b(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_bianjie0(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    PointCloudBoundary2(cloud_b_a, cloud_b_b);
    peidui(cloud_new, cloud_b_b, cloud_bianjie0);

    // 第二次边界检测与配对
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a1(new pcl::PointCloud<pcl::PointXYZ>());
    cloud_a1->resize(cloud_new->size());
    for (size_t i = 0; i < cloud_new->size(); i++) {
        (*cloud_a1)[i].x = cloud_new->points[i].x;
        (*cloud_a1)[i].y = cloud_new->points[i].y;
        (*cloud_a1)[i].z = cloud_new->points[i].z;
    }
    cloud_a1->width = static_cast<uint32_t>(cloud_a1->points.size());
    cloud_a1->height = 1;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_b1(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_c1(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    PointCloudBoundary2(cloud_a1, cloud_b1);
    peidui(cloud_new, cloud_b1, cloud_c1);

    float cvs = 0.0f;
    CVS(cloud_new, cloud_c1, cloud_bianjie0, cvs);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_bj(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    cloud_bj->resize(cloud_c1->size());
    for (size_t i = 0; i < cloud_c1->size(); i++) {
        (*cloud_bj)[i].x = cloud_c1->points[i].x;
        (*cloud_bj)[i].y = cloud_c1->points[i].y;
        (*cloud_bj)[i].z = cloud_c1->points[i].z;
        (*cloud_bj)[i].r = cloud_c1->points[i].r;
        (*cloud_bj)[i].g = cloud_c1->points[i].g;
        (*cloud_bj)[i].b = cloud_c1->points[i].b;
    }
    cloud_bj->width = static_cast<uint32_t>(cloud_bj->points.size());
    cloud_bj->height = 1;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_w_a(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_w_b(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_w_bianjie0(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    // 控制迭代次数，避免死循环；当 cvs >= 1.0 代表完全匹配，这里按原意改为当 cvs < 1.0 继续迭代
    int max_iters = 5;
    for (int iter = 0; iter < max_iters && cvs >= 1.0f; ++iter) {
        outputClouds.clear();
        outputClouds_w.clear();
        seedIndices.clear();
        findCommonPoints(cloud_new, cloud_bj, seedIndices);

        regionGrowing_RGB(cloud_new, seedIndices, distanceThreshold, normalThreshold, qlThreshold, rgbThreshold,
                          normals, outputClouds, outputClouds_w, minRegionSize, maxAdditionalSeeds, km, k_1);

        cloud_new->clear();
        if (!outputClouds_w.empty() && outputClouds_w[0]) {
            liqun(outputClouds_w[0], cloud_new);
        }

        cloud_w_a->resize(cloud_new->size());
        for (size_t i = 0; i < cloud_new->size(); i++) {
            (*cloud_w_a)[i].x = cloud_new->points[i].x;
            (*cloud_w_a)[i].y = cloud_new->points[i].y;
            (*cloud_w_a)[i].z = cloud_new->points[i].z;
        }
        cloud_w_a->width = static_cast<uint32_t>(cloud_w_a->points.size());
        cloud_w_a->height = 1;

        PointCloudBoundary2(cloud_w_a, cloud_w_b);
        peidui(cloud_new, cloud_w_b, cloud_w_bianjie0);

        cloud_bj->clear();
        cloud_bj->resize(cloud_w_bianjie0->size());
        for (size_t i = 0; i < cloud_w_bianjie0->size(); i++) {
            (*cloud_bj)[i].x = cloud_w_bianjie0->points[i].x;
            (*cloud_bj)[i].y = cloud_w_bianjie0->points[i].y;
            (*cloud_bj)[i].z = cloud_w_bianjie0->points[i].z;
            (*cloud_bj)[i].r = cloud_w_bianjie0->points[i].r;
            (*cloud_bj)[i].g = cloud_w_bianjie0->points[i].g;
            (*cloud_bj)[i].b = cloud_w_bianjie0->points[i].b;
        }
        cloud_bj->width = static_cast<uint32_t>(cloud_bj->points.size());
        cloud_bj->height = 1;

        CVS(cloud_new, cloud_w_bianjie0, cloud_bianjie0, cvs);
        cloud_w_a->clear();
        cloud_w_b->clear();
        cloud_w_bianjie0->clear();
    }

    // 保存输出
    if (!cloud_new->empty()) {
        pcl::io::savePCDFileBinary("D:\\Desktop\\LS\\cloud_new.pcd", *cloud_new);
        std::cout << "Saved: D:\\\\Desktop\\\\LS\\\\out\\\\cloud_new.pcd" << std::endl;
    } else {
        std::cout << "cloud_new is empty; nothing saved." << std::endl;
    }

    return 0;
}

