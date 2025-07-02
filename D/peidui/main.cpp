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
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_A (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_B (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_C (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\mox1\\quchuliqundian_8.0.pcd", *cloud_A) == -1) {
        PCL_ERROR("Couldn't read file cloud.pcd\n");
        return -1;
    }
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("D:\\Desktop\\LS\\mox1\\bj_8.0.pcd", *cloud_B) == -1) {
        PCL_ERROR("Couldn't read file cloud.pcd\n");
        return -1;
    }

    for(const auto& point_a : cloud_A->points){
        for(const auto point_b : cloud_B->points){
            if (point_a.x == point_b.x && point_a.y == point_b.y && point_a.z == point_b.z
                ) {

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
    pcl::io::savePCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\mox1\\bb_8.0.pcd", *cloud_C);

    std::cout << "Filtered cloud saved with " << cloud_C->points.size() << " points." << std::endl;

    return 0;

}
