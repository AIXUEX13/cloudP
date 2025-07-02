#include <iostream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>

int main(int argc, char** argv) {
    // 读取点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_A(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_C(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\mox1\\wfgbf_x_0.pcd", *cloud_A) == -1) {
        PCL_ERROR("Couldn't read file cloud.pcd\n");
        return -1;
    }
    cloud->resize(cloud_A->points.size());
    for (size_t i = 0; i < cloud_A->points.size(); ++i) {
        cloud->points[i].x = cloud_A->points[i].x;
        cloud->points[i].y = cloud_A->points[i].y;
        cloud->points[i].z = cloud_A->points[i].z;
        cloud->points[i].r = cloud_A->points[i].r;
        cloud->points[i].g = cloud_A->points[i].g;
        cloud->points[i].b = cloud_A->points[i].b;
    }

    // 创建KD树搜索对象
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud);

    // 创建欧几里得聚类对象
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    std::vector<pcl::PointIndices> cluster_indices; // 存储各个簇的点索引
    ec.setClusterTolerance(0.3);  // 设置聚类的容忍度（单位：米）
    ec.setMinClusterSize(2000);     // 设置簇的最小点数
    ec.setMaxClusterSize(10000000);   // 设置簇的最大点数（可选）
    ec.setSearchMethod(tree);      // 设置搜索方式为KD树
    ec.setInputCloud(cloud);       // 设置输入点云

    // 执行聚类
    ec.extract(cluster_indices);

    std::cout << "Number of clusters: " << cluster_indices.size() << std::endl;

    // 新建点云对象保存去除小簇后的点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_final(new pcl::PointCloud<pcl::PointXYZRGB>());

    // 遍历每个簇，检查大小并过滤掉小簇
    for (const auto& indices : cluster_indices) {
        if (indices.indices.size() >= 2000) { // 只保留大小大于等于100的簇
            for (const auto& index : indices.indices) {
                cloud_filtered_final->points.push_back(cloud->points[index]);
            }
        }
    }

    cloud_filtered_final->width = cloud_filtered_final->points.size();
    cloud_filtered_final->height = 1;

    for (const auto& point_b : cloud_filtered_final->points) {
        for (const auto& point_a : cloud_A->points) {
            // 比较 a 中的 point 和 b 中的 point 的 xyzrgb
            if (point_a.x == point_b.x && point_a.y == point_b.y && point_a.z == point_b.z &&
                point_a.r == point_b.r && point_a.g == point_b.g && point_a.b == point_b.b) {

                // 检查是否已存在于 cloud_C 中
                bool point_exists = false;
                for (const auto& existing_point : cloud_C->points) {
                    if (existing_point.x == point_a.x && existing_point.y == point_a.y && existing_point.z == point_a.z &&
                        existing_point.r == point_a.r && existing_point.g == point_a.g && existing_point.b == point_a.b) {
                        point_exists = true;
                        break;
                    }
                }

                // 如果点不存在于 cloud_C 中，则复制到 cloud_C
                if (!point_exists) {
                    pcl::PointXYZRGBNormal point_c;
                    point_c.x = point_a.x;
                    point_c.y = point_a.y;
                    point_c.z = point_a.z;
                    point_c.r = point_a.r;
                    point_c.g = point_a.g;
                    point_c.b = point_a.b;
                    point_c.normal_x = point_a.normal_x;
                    point_c.normal_y = point_a.normal_y;
                    point_c.normal_z = point_a.normal_z;

                    cloud_C->points.push_back(point_c);
                }
            }
        }
    }

    cloud_C->width = cloud_C->points.size();
    cloud_C->height = 1;
    // 保存去除小簇后的点云
    pcl::io::savePCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\mox1\\quchuliqundian_8.0.pcd", *cloud_C);
    std::cout <<"cloud_a.size="<<cloud_A->points.size()<<std::endl;
    std::cout <<"cloud_b.size="<<cloud_filtered_final->points.size()<<std::endl;
    std::cout << "Filtered cloud saved with " << cloud_C->points.size() << " points." << std::endl;

    return 0;
}
