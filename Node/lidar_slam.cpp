#define PCL_NO_PRECOMPILE
#include <mutex>
#include <queue>
#include <fstream>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseStamped.h>
#include "algorithm/lidarOdom.h"
#include "utils/KalmanFilter.h"
#include "utils/tic_toc.h"
using namespace std;
using namespace lom;
using namespace art;

Eigen::Affine3d transf_se_ = Eigen::Affine3d::Identity();
Eigen::Affine3d transf_sum_ = Eigen::Affine3d::Identity();
Eigen::Affine3d transf_sum_last_ = Eigen::Affine3d::Identity();
Eigen::Affine3d orig_pose = Eigen::Affine3d::Identity();

nav_msgs::Odometry laser_odom_last;

std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;

std::mutex mBuf;

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_slam");
    ros::NodeHandle nh("~");

    bool first_frame = true;

    bool using_PointXYZI = true;
    bool using_PointXYZIR = true;
    bool using_PointXYZIRT = false;

    string strSettingsFile = "/home/lutao/Documents/my_slam_ws/src/lt_slam/cfg/lidar_slam.yaml";
    cv::FileStorage fSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fSettings.isOpened()){
        std::cout << "Failed to open settings file at: " << strSettingsFile << std::endl;
        return 0 ;
    }else{
        std::cout << "Success to open settings file at: " << strSettingsFile << std::endl;
    }
    OdomConfig config;
//    config.SetupParam(nh);
    config.setParam(fSettings);
    Odom odom(config, nh);

    LOG(INFO) << "OK";

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1000, laserCloudFullResHandler);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/localization_test_laser_odom", 100);
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/localization_test_laser_path", 100);
    ros::Publisher pubMapCloud = nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 100);
    tf::TransformBroadcaster tf_br_;

    nav_msgs::Path laser_path;

    int frameCount = 0;
    ros::Rate rate(2000);

    while (ros::ok()) {
        ros::spinOnce();
        if (!fullPointsBuf.empty()) {
            ros::Time start_time = ros::Time::now();

            mBuf.lock();
            sensor_msgs::PointCloud2 source_cloud;
            source_cloud = *fullPointsBuf.front();
            fullPointsBuf.pop();
            mBuf.unlock();

            if (first_frame) {
                transf_se_.setIdentity();
            }

            PointIRTCloud laser_cloud_out;
            if (using_PointXYZI) {
                PointICloud laser_cloud_in;
                pcl::fromROSMsg(source_cloud, laser_cloud_in);
                vector<int> ind;
                laser_cloud_in.is_dense = false;
                pcl::removeNaNFromPointCloud(laser_cloud_in, laser_cloud_in, ind);
                CalculateRingAndTime(laser_cloud_in, laser_cloud_out);
//                odom.EstimatePoseForSLAM(laser_cloud_out, transf_sum_, transf_se_);

                sensor_msgs::PointCloud2 ring_cloud;
                pcl::toROSMsg(laser_cloud_out, ring_cloud);
                ring_cloud.header.stamp = ros::Time::now();
                ring_cloud.header.frame_id = "/velodyne";
                pubMapCloud.publish(ring_cloud);

                PointIRTCloud feature_cloud;
                odom.GetFeaturePoints(laser_cloud_out, feature_cloud);
            } else if (using_PointXYZIR) {
                PointIRCloud laser_cloud_in;
                pcl::fromROSMsg(source_cloud, laser_cloud_in);
                vector<int> ind;
                laser_cloud_in.is_dense = false;
                pcl::removeNaNFromPointCloud(laser_cloud_in, laser_cloud_in, ind);
                CalculateRingAndTime(laser_cloud_in, laser_cloud_out);
                odom.EstimatePoseForSLAM(laser_cloud_out, transf_sum_, transf_se_);
            } else if (using_PointXYZIRT) {
                PointIRTCloud laser_cloud_in;
                pcl::fromROSMsg(source_cloud, laser_cloud_in);
                vector<int> ind;
                pcl::removeNaNFromPointCloud(laser_cloud_in, laser_cloud_in, ind);           
                CalculateRingAndTime(laser_cloud_in, laser_cloud_out);
                odom.EstimatePoseForSLAM(laser_cloud_out, transf_sum_, transf_se_);
            }

            if (first_frame) {
                first_frame = false;
                transf_se_ = Eigen::Affine3d::Identity();
            } else {
                transf_se_ = transf_sum_last_.inverse() * transf_sum_;
            }

            transf_sum_last_ = transf_sum_;

            ros::Time time_now = ros::Time::now();
            double off_time = time_now.toSec() - start_time.toSec();
            std::cout << "time_1: " << off_time * 1000 << "ms" << std::endl << std::endl << std::endl;

            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/velodyne";
            laserOdometry.header.stamp = ros::Time::now();
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.pose.pose.position.x = transf_sum_.translation().x();
            laserOdometry.pose.pose.position.y = transf_sum_.translation().y();
            laserOdometry.pose.pose.position.z = transf_sum_.translation().z();
            laserOdometry.pose.pose.orientation.x = Eigen::Quaterniond(transf_sum_.rotation()).x();
            laserOdometry.pose.pose.orientation.y = Eigen::Quaterniond(transf_sum_.rotation()).y();
            laserOdometry.pose.pose.orientation.z = Eigen::Quaterniond(transf_sum_.rotation()).z();
            laserOdometry.pose.pose.orientation.w = Eigen::Quaterniond(transf_sum_.rotation()).w();
            pubLaserOdometry.publish(laserOdometry);

            // publish laser path
            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laser_path.poses.push_back(laserPose);
            laser_path.header = laserOdometry.header;
            pubLaserPath.publish(laser_path);

            tf::Transform transform;
            tf::Quaternion tfq(laserOdometry.pose.pose.orientation.x,laserOdometry.pose.pose.orientation.y,
                               laserOdometry.pose.pose.orientation.z,laserOdometry.pose.pose.orientation.w);
            transform.setOrigin(tf::Vector3(transf_sum_.translation().x(),
                                            transf_sum_.translation().y(),
                                            transf_sum_.translation().z()));
            transform.setRotation(tfq);
            tf_br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/velodyne", "/now_position"));
        }
        rate.sleep();
    }
    return 0;
}
