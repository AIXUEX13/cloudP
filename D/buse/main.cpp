#include <iostream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>

int main(int argc, char *argv[])
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_A(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_B(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_C(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\YK_pxh.pcd", *cloud_A) == -1) {
        PCL_ERROR("Couldn't read file cloud.pcd\n");
        return -1;
    }

    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\3.13\\qie.pcd", *cloud_B) == -1) {
        PCL_ERROR("Couldn't read file cloud.pcd\n");
        return -1;
    }





    for (const auto& point_b : cloud_B->points) {
        for (auto& point_a : cloud_A->points) {
            // 比较 a 中的 point 和 b 中的 point 的 xyzrgb
            if (point_a.x == point_b.x && point_a.y == point_b.y && point_a.z == point_b.z
                ) {
                point_a.rgba = point_b.rgba;


            }
        }
    }

    // 保存去除小簇后的点云
    pcl::io::savePCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\3.13\\bianse_1_0.pcd", *cloud_A);

    std::cout << "Filtered cloud saved with " << cloud_A->points.size() << " points." << std::endl;

    return 0;
}
