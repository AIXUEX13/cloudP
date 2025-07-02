#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/principal_curvatures.h>
#include <vector>
#include <algorithm>
#include <pcl/io/pcd_io.h>  // 需要包含这个头文件来保存PCD文件
#include <fstream>  // 用于保存到文本文件

struct PointWithCurvature {
    pcl::PointXYZRGBNormal point;
    float curvature;

    bool operator<(const PointWithCurvature& other) const {
        return curvature < other.curvature;  // 根据曲率的绝对值排序
    }
};
std::stringstream ss;

void computeCurvaturesAndSort(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudXYZRGBNormal,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudXYZSorted
                            ) {
    // Step 1: 创建一个新的点云（pcl::PointXYZ）并将XYZ坐标从原始点云复制过来
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
    for (size_t i = 0; i < cloudXYZRGBNormal->points.size(); ++i) {
        pcl::PointXYZ point;
        point.x = cloudXYZRGBNormal->points[i].x;
        point.y = cloudXYZRGBNormal->points[i].y;
        point.z = cloudXYZRGBNormal->points[i].z;
        cloudXYZ->points.push_back(point);
    }

    // Step 2: 计算主曲率
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> curvatureEstimation;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // 计算法向量（在此代码中假设法向量已经计算并存储在cloudXYZRGBNormal中）
    for (size_t i = 0; i < cloudXYZRGBNormal->points.size(); ++i) {
        pcl::Normal normal;
        normal.normal_x = cloudXYZRGBNormal->points[i].normal_x;
        normal.normal_y = cloudXYZRGBNormal->points[i].normal_y;
        normal.normal_z = cloudXYZRGBNormal->points[i].normal_z;
        normals->points.push_back(normal);
    }

    // 使用点云和法向量计算主曲率
    pcl::PointCloud<pcl::PrincipalCurvatures> curvatures;
    curvatureEstimation.setInputCloud(cloudXYZ);
    curvatureEstimation.setInputNormals(normals);

    float radius = 1.2f;
    curvatureEstimation.setRadiusSearch(radius);
    curvatureEstimation.compute(curvatures);

    // Step 3: 存储每个点及其对应的曲率值
    std::vector<PointWithCurvature> pointsWithCurvature;
    for (size_t i = 0; i < cloudXYZ->points.size(); ++i) {
        PointWithCurvature pwc;
        pwc.point = cloudXYZRGBNormal->points[i];
        // 选择曲率的绝对值最小的一个（pc1 和 pc2 的绝对值）
        pwc.curvature = std::min(fabs(curvatures.points[i].pc1), fabs(curvatures.points[i].pc2));  // 取绝对值较小的曲率
        pointsWithCurvature.push_back(pwc);
    }

    // Step 4: 按照曲率的绝对值对点进行排序
    std::sort(pointsWithCurvature.begin(), pointsWithCurvature.end());

    // Step 5: 根据排序结果重建点云（按曲率排序）
    cloudXYZRGBNormal->points.clear();  // 清空点云以重新填充
    for (const auto& pwc : pointsWithCurvature) {
        cloudXYZRGBNormal->points.push_back(pwc.point);
    }

    // Step 6: 保存排序后的点云为PCD文件
    pcl::io::savePCDFileASCII(ss.str(), *cloudXYZRGBNormal);
    std::cout << "Sorted point cloud saved to 'sorted_cloud.pcd'." << std::endl;

    // Step 7: 保存排序后的点云数据到文本文件
    std::ofstream outfile("D:\\Desktop\\LS\\4.22\\pxh_.csv");
    for (const auto& pwc : pointsWithCurvature) {
    //     // 写入每个点的xyz, rgb, normal和曲率值
        outfile << pwc.curvature<<std::endl;
    }
    //             << static_cast<int>(pwc.point.r) << "," << static_cast<int>(pwc.point.g) << "," << static_cast<int>(pwc.point.b) << ","
    //             << pwc.point.normal_x << "," << pwc.point.normal_y << "," << pwc.point.normal_z << ","
    //             << pwc.curvature << std::endl;
    // }
    // outfile.close();
    // std::cout << "Curvature data saved to 'pxh.csv'." << std::endl;
}

int main() {
    // 假设你已经有一个点云对象 cloudXYZRGBNormal
    std::stringstream ss1;
    std::string s1="D:\\Desktop\\LS\\4.22\\mox3\\chusd";
    ss << s1<<"_"  << ".pcd";
    ss1<<s1<<".pcd";
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudXYZRGBNormal(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal> (ss1.str(), *cloudXYZRGBNormal) == -1) { //* load the file
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }
    // 调用计算曲率并排序的函数
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZSorted(new pcl::PointCloud<pcl::PointXYZ>());
    computeCurvaturesAndSort(cloudXYZRGBNormal, cloudXYZSorted);

    // 现在 cloudXYZRGBNormal 包含了按曲率排序的点云，并已保存为 'sorted_cloud.pcd'
    return 0;
}
