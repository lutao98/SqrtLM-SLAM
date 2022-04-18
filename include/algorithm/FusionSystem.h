#ifndef FUSIONSYSTEM_H
#define FUSIONSYSTEM_H
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cv_bridge/cv_bridge.h>

#include <Eigen/Core>

#include <vector>

#include "System.h"
#include "utils/common.h"
#include "utils/tic_toc.h"
#include "utils/lidarconfig.h"


namespace LT_SLAM
{

const float r = 5;
const int half_patch_width=4;
const int half_patch_height=7;

class FusionSystem{
private:

    std::mutex mtx_;

    lidarConfig lidarconfig_;

    string ImgFilePath_;

    ros::Subscriber pointCloudSub_;

    ros::Publisher image_pub_;
    ros::Publisher depthimage_pub_;
    ros::Publisher rangeimage_pub_;
    ros::Publisher rotated_pc_pub_;
    ros::Publisher color_pc_pub_;
    ros::Publisher depthline_pub_;
    ros::Publisher map_pub_;
    ros::Publisher localmap_pub_;
    ros::Publisher groundtruth_pub_;
    ros::Publisher SLAMpath_pub_;
    ros::Publisher ORBSLAMstereopath_pub_;
    ros::Publisher ORBSLAMmonopath_pub_;
    ros::Publisher KFmarker_pub_;
    ros::Publisher sharp_cloud_pub_;
    ros::Publisher flat_cloud_pub_;
    ros::Publisher less_sharp_cloud_pub_;
    ros::Publisher less_flat_cloud_pub_;
    ros::Publisher lidar_local_map_pub_;

    nav_msgs::Path GTpath_;
    nav_msgs::Path SLAMpath_;
    nav_msgs::Path ORBSLAMstereopath_;
    nav_msgs::Path ORBSLAMmonopath_;

    tf::TransformBroadcaster tf_br_;

    visualization_msgs::MarkerArray KeyFrameVisual_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_inputPtr_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr lidar_colorPtr_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr lidar_colorMapPtr_;
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_;

    cv::Mat Tcw_;
    cv::Mat img_gray_;
    std::vector<double> vTimestamps_;

    Eigen::Matrix<double,3,4> intrinsicMatrix_;  //3x4 projection matrix after rectification, P_rect_02:3x4
    Eigen::Matrix<double,4,4> extrinsicMatrix_;  //Transform from velo to cam0, T:4x4

    unsigned int frameNum_ = 0;
    unsigned int view_framenum_ = 0;

    void callback(const sensor_msgs::PointCloud2::ConstPtr &PointCloudMsg);

//    // 可修改
//    Eigen::Vector3i TransformProject(const Eigen::Vector4d &P_lidar)
//    {
//        Eigen::Vector3d z_P_uv = intrinsicMatrix_*extrinsicMatrix_*P_lidar;
//        return Eigen::Vector3i( int( z_P_uv[0]/z_P_uv[2] ), int( z_P_uv[1]/z_P_uv[2] ), 1 );
//    }

    std::string GetFrameStr(unsigned int frame)
    {
        if(frame>9999)
            return "0"+std::to_string(frame);
        else if(frame>999)
            return "00"+std::to_string(frame);
        else if(frame>99)
            return "000"+std::to_string(frame);
        else if(frame>9)
            return "0000"+std::to_string(frame);
        else if(frame<=9)
            return "00000"+std::to_string(frame);
    }

public:

    ORB_SLAM2::System SLAM_;

    FusionSystem(ros::NodeHandle &nh, cv::FileStorage &cfg);

    ~FusionSystem(){};

    void Spin(){
        ros::spin();
    };

    void Visualization();

    void pubMessage();

};

}

#endif // FUSIONSYSTEM_H
