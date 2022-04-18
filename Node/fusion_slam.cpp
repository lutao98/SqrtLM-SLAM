#include "algorithm/FusionSystem.h"
#include "algorithm/lidarOdom.h"
#include "utils/KalmanFilter.h"
#include "utils/tic_toc.h"
int main( int argc, char** argv)
{
    ros::init(argc, argv, "calibrate");
    ros::NodeHandle nh;

    string strSettingsFile = "/home/lutao/Documents/my_slam_ws/src/lt_slam/cfg/KITTIpath.yaml";
    cv::FileStorage fSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fSettings.isOpened()){
        std::cout << "Failed to open settings file at: " << strSettingsFile << std::endl;
        return 0 ;
    }else{
        std::cout << "Success to open settings file at: " << strSettingsFile << std::endl;
    }

    LT_SLAM::FusionSystem fusion_system(nh,fSettings);
    std::thread loopthread(&LT_SLAM::FusionSystem::Visualization, &fusion_system);

    fusion_system.Spin();

    loopthread.join();
//    loopthread.detach();
    return 0;
}
