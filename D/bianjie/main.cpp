#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <chrono>

int PointCloudBoundary2(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // 1. 计算法向量
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setSearchMethod(tree);
    normalEstimation.setRadiusSearch(1);  // 法向量搜索半径，调整以适应路面点云密度
    normalEstimation.compute(*normals);      // 计算法向量

    // 2. 边界估计
    pcl::PointCloud<pcl::Boundary>::Ptr boundaries(new pcl::PointCloud<pcl::Boundary>);
    boundaries->resize(cloud->size());  // 初始化大小
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimation;
    boundary_estimation.setInputCloud(cloud);
    boundary_estimation.setInputNormals(normals);  // 设置法线
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_ptr(new pcl::search::KdTree<pcl::PointXYZ>);
    boundary_estimation.setSearchMethod(kdtree_ptr);  // 设置搜索方式
    boundary_estimation.setKSearch(30);               // 设置K近邻，适应稠密点云
    boundary_estimation.setAngleThreshold(M_PI * 0.5);  // 角度阈值，大于该值为边界点
    boundary_estimation.compute(*boundaries);          // 计算边界

    // 统计边界点数量
    int boundary_count = 0;
    for (size_t i = 0; i < boundaries->size(); i++)
    {
        if (boundaries->points[i].boundary_point != 0)
        {
            boundary_count++;
        }
    }
    std::cout << "边界点的数量: " << boundary_count << std::endl;

    // 3. 可视化和保存
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_visual(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_boundary(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_visual->resize(cloud->size());

    for (size_t i = 0; i < cloud->size(); i++)
    {
        cloud_visual->points[i].x = cloud->points[i].x;
        cloud_visual->points[i].y = cloud->points[i].y;
        cloud_visual->points[i].z = cloud->points[i].z;
        if (boundaries->points[i].boundary_point != 0)  // 如果是边界点
        {
            cloud_visual->points[i].r = 255;
            cloud_visual->points[i].g = 0;
            cloud_visual->points[i].b = 0;
            cloud_boundary->push_back(cloud_visual->points[i]);  // 保存边界点
        }
        else  // 非边界点
        {
            cloud_visual->points[i].r = 255;
            cloud_visual->points[i].g = 255;
            cloud_visual->points[i].b = 255;
        }
    }

    // 保存结果
    //pcl::io::savePCDFileBinaryCompressed("D:\\Desktop\\LS\\mox3\\bianjie_all_0.0.pcd", *cloud_visual);
    pcl::io::savePCDFileBinaryCompressed("D:\\Desktop\\LS\\mox3\\yige.0.pcd", *cloud_boundary);

    return 0;
}

void visualizePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);  // 设置背景颜色为黑色
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem(1.0);  // 添加坐标系
    viewer->initCameraParameters();

    // 循环显示直到窗口关闭
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char** argv)
{
    // 加载点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("D:\\Desktop\\LS\\mox3\\yige.pcd", *cloud) == -1)
    {
        PCL_ERROR("无法加载点云文件\n");
        return (-1);
    }

    // 调用边界提取函数
    PointCloudBoundary2(cloud);

    // 加载带颜色的可视化点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_visual(new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::io::loadPCDFile<pcl::PointXYZRGB>("D:\\Desktop\\LS\\mox3\\yige_bj.0.pcd", *cloud_visual);

    // 调用可视化函数
   // visualizePointCloud(cloud_visual);

    return 0;
}

