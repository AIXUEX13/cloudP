#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <chrono>
#include <vector>
#include <pcl/visualization/pcl_visualizer.h>

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

int main(int argc, char *argv[])
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_C (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\mox1\\bb_1.0.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file input.pcd\n");
        return -1;
    }
    // pcl::PointXYZRGBNormal first_point = cloud->points[999];
    // float r_f = first_point.r;
    // float g_f = first_point.g;
    // float b_f = first_point.b;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_A (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\mox1\\quchuliqundian_8.0.pcd", *cloud_A) == -1) {
        PCL_ERROR("Couldn't read file input.pcd\n");
        return -1;
    }
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_Z (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\mox1\\bb_8.0.pcd", *cloud_Z) == -1) {
        PCL_ERROR("Couldn't read file input.pcd\n");
        return -1;
    }
    float num=0;
    for (const auto& point_a : cloud_A->points){
        for(const auto& point_b :cloud->points){
        pcl::PointXYZRGBNormal point_c;
        float ra,ga,ba,rb,gb,bb;
        ra = point_a.r;
        ga = point_a.g;
        ba = point_a.b;

        rb = point_b.r;
        gb = point_b.g;
        bb = point_b.b;
        float co;
        co = sqrt((ra - rb) * (ra - rb) +(ga - gb) * (ga - gb) + (ba - bb) * (ba - bb) );
        if(co <= 1.5){
            num++;
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
            break;
         }
        }


    }

    cloud_C->width = cloud_C->points.size();
    cloud_C->height = 1;

    pcl::io::savePCDFile<pcl::PointXYZRGBNormal>("D:\\Desktop\\LS\\mox2\\qianfenge_4.pcd", *cloud_C);

    float cvs = num/cloud_Z->size();
    cout<<"cvs:"<<cvs*100<<"%"<<endl;
    cout<<"num:"<<num<<endl;

    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_visual(new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::io::loadPCDFile<pcl::PointXYZRGB>("D:\\Desktop\\LS\\4.9\\qianfenge.pcd", *cloud_visual);

    // // 调用可视化函数
    //visualizePointCloud(cloud_visual);

}
