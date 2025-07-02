#ifndef JCGS_H
#define JCGS_H

#include <cmath>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
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

float Radian_D (const pcl::PointXYZRGBNormal& a, const pcl::PointXYZRGBNormal& b){
    double dot_product = a.normal_x * b.normal_x + a.normal_y * b.normal_y + a.normal_z * b.normal_z;
    return std::acos(dot_product) * 180.0/M_PI;
}

#endif // JCGS_H
