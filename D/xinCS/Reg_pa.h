#ifndef REG_PA_H
#define REG_PA_H
#include <iostream>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/search/kdtree.h>

void Reg_f(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    // 创建 KDTree 搜索结构
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>());

    // 设置主曲率估计器
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZRGBNormal, pcl::Normal, pcl::PrincipalCurvatures> pc_est;
    pc_est.setInputCloud(cloud);
    pc_est.setInputNormals(normals);
    pc_est.setSearchMethod(tree);
    pc_est.setKSearch(50);  // 使用 50 个邻居来估计曲率

    // 存储计算得到的主曲率
    pcl::PointCloud<pcl::PrincipalCurvatures> principal_curvatures;
    pc_est.compute(principal_curvatures);

    // 将曲率和对应点的索引存储在 vector 中，以便排序
    std::vector<std::pair<float, int>> curvature_data;

    for (int i = 0; i < cloud->size(); ++i) {
        // 获取每个点的曲率数据
        const pcl::PrincipalCurvatures& curv = principal_curvatures.points[i];
        // 使用最大的曲率作为排序依据（可以使用主曲率的两个分量中的任意一个，这里选择最大的）
        float max_curvature = std::max(curv.pc1, curv.pc2);  // pc1 和 pc2 是主曲率
        curvature_data.push_back(std::make_pair(max_curvature, i));
    }

    // 按曲率排序，降序排列（曲率大的在前面）
    std::sort(curvature_data.begin(), curvature_data.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first; // 通过曲率的值进行排序
              });

    // 创建一个新的点云对象，用于存储排序后的点云
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sorted_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    // 根据排序后的索引重组点云
    for (const auto& pair : curvature_data) {
        int idx = pair.second;
        sorted_cloud->points.push_back(cloud->points[idx]);
    }

    // 保存排序后的点云为 PCD 文件
    pcl::io::savePCDFile("sorted_by_curvature.pcd", *sorted_cloud);

    // 输出排序后的结果
    std::cout << "Sorted points by curvature saved to 'sorted_by_curvature.pcd'." << std::endl;
    for (const auto& pair : curvature_data) {
        int idx = pair.second;
        float curvature = pair.first;
        std::cout << "Point index: " << idx << ", Curvature: " << curvature << std::endl;
    }
}

#endif // REG_PA_H
