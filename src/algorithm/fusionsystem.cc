#include "algorithm/FusionSystem.h"

LT_SLAM::FusionSystem::FusionSystem(ros::NodeHandle &nh, cv::FileStorage &cfg):SLAM_(cfg["File.ORBvoc"],
                                                                                     cfg["File.KITTI"],
                                                                                     ORB_SLAM2::System::RGBD,
                                                                                     false)
{

    int loop_detection=cfg["SLAM.LoopDetection"];
    SLAM_.setLoopDetection(loop_detection);

    std::cout << std::endl << "Loop Detection:" << loop_detection << std::endl << std::endl;

    pointCloudSub_ = nh.subscribe("/velodyne_points", 1000 , &FusionSystem::callback, this);

    image_pub_ = nh.advertise<sensor_msgs::Image> ("/lidar_image", 1);
    depthimage_pub_ = nh.advertise<sensor_msgs::Image> ("/lidardepth_image", 1);
    rangeimage_pub_ = nh.advertise<sensor_msgs::Image> ("/range_image", 1);

    rotated_pc_pub_ = nh.advertise<sensor_msgs::PointCloud2> ("/rotated_lidar", 1);
    color_pc_pub_ = nh.advertise<sensor_msgs::PointCloud2> ("/color_lidar", 1);
    depthline_pub_ = nh.advertise<sensor_msgs::PointCloud2> ("/depth_line", 1);
    map_pub_ = nh.advertise<sensor_msgs::PointCloud2> ("/map", 1);
    localmap_pub_ = nh.advertise<sensor_msgs::PointCloud2> ("/local_map", 1);

    groundtruth_pub_ = nh.advertise<nav_msgs::Path>("/GT_path", 1);
    SLAMpath_pub_ = nh.advertise<nav_msgs::Path>("/SLAMpath_", 1);
    ORBSLAMstereopath_pub_ = nh.advertise<nav_msgs::Path>("/ORBSLAM_stereo_path_", 1);
    ORBSLAMmonopath_pub_ = nh.advertise<nav_msgs::Path>("/ORBSLAM_mono_path_", 1);
    KFmarker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/KF_marker", 1);

    sharp_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/sharp_cloud", 100);
    flat_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/flat_cloud", 100);
    less_sharp_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/less_sharp_cloud", 100);
    less_flat_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/less_flat_cloud", 100);
    lidar_local_map_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/lidar_local_map", 100);

    lidar_inputPtr_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    lidar_colorPtr_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    lidar_colorMapPtr_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    string lidarconfig_strSettings = cfg["File.lidar_config"];
    cv::FileStorage lidarfSetting(lidarconfig_strSettings.c_str(), cv::FileStorage::READ);
    if(!lidarfSetting.isOpened()){
        std::cout << "Failed to open settings file at: " << lidarconfig_strSettings << std::endl;
    }else{
        std::cout << "Success to open settings file at: " << lidarconfig_strSettings << std::endl;
    }
    lidarconfig_.setParam(lidarfSetting);
    SLAM_.setLidarConfig(&lidarconfig_);

    ifstream fCalib;
    string strCalibFile = cfg["File.calib"];
    fCalib.open(strCalibFile.c_str());
    std::vector<double> invec,exvec;
    while(!fCalib.eof()){
        string s;
        getline(fCalib,s);
        if(s.substr(0,2)=="P0"){
            string P0=s.substr(4,s.size()-1);
            stringstream ss;
            ss << P0;
            while(!ss.eof()){
                double num;
                ss >> num;
                invec.emplace_back(num);
            }
        }else if(s.substr(0,2)=="Tr"){
            string Tr=s.substr(4,s.size()-1);
            stringstream ss;
            ss << Tr;
            while(!ss.eof()){
                double num;
                ss >> num;
                exvec.emplace_back(num);
            }
        }
    }

    intrinsicMatrix_ << invec.at(0), invec.at(1), invec.at(2), invec.at(3),
                        invec.at(4), invec.at(5), invec.at(6), invec.at(7),
                        invec.at(8), invec.at(9), invec.at(10), invec.at(11);
    extrinsicMatrix_ << exvec.at(0), exvec.at(1), exvec.at(2), exvec.at(3),
                        exvec.at(4), exvec.at(5), exvec.at(6), exvec.at(7),
                        exvec.at(8), exvec.at(9), exvec.at(10), exvec.at(11),
                        0, 0, 0, 1;

    SLAM_.setCalibrationParam(intrinsicMatrix_,extrinsicMatrix_);

    std::cout << "Intrinsics Matrix:\n" << intrinsicMatrix_ << std::endl
              << "Calibration Matrix:\n" << extrinsicMatrix_ << std::endl;

    string ImgPath = cfg["File.Img"];
    ImgFilePath_=ImgPath;

    ifstream fTimes;
    string strPathTimeFile = cfg["File.times"];
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof()){
        string s;
        getline(fTimes,s);
        // 当该行不为空的时候执行
        if(!s.empty()){
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            // 保存时间戳
            vTimestamps_.emplace_back(t);
        }
    }
    std::cout << std::endl << "timestamps has frame: " << vTimestamps_.size() << std::endl << std::endl;

    ifstream fGT;
    string strGTFile = cfg["File.Groundtruth"];
    fGT.open(strGTFile.c_str());
    std::vector<Eigen::Vector4d> GT;
    while(!fGT.eof())
    {
        double row[12];
        fGT >> row[0] >> row[1] >> row[2] >> row[3] >> row[4] >> row[5]
            >> row[6] >> row[7] >> row[8] >> row[9] >> row[10] >> row[11] ;
        Eigen::Vector4d position=extrinsicMatrix_.inverse()*(Eigen::Vector4d(row[3],row[7],row[11],1));
        GT.emplace_back(position);
    }

    for(int i=0;i<GT.size();i++){
        geometry_msgs::PoseStamped this_pose_stamped;
        this_pose_stamped.pose.position.x = GT.at(i).x();
        this_pose_stamped.pose.position.y = GT.at(i).y();
        this_pose_stamped.pose.position.z = GT.at(i).z();
        this_pose_stamped.header.stamp=ros::Time::now();
        this_pose_stamped.header.frame_id="fusion_slam";
        GTpath_.poses.emplace_back(this_pose_stamped);
    }
    GTpath_.header.stamp = ros::Time::now();
    GTpath_.header.frame_id = "fusion_slam";

    ifstream fORBSLAM_stereo;
    string strORBSLAM_stereoFile = cfg["File.stereo"];     // 上一次跑的轨迹
    fORBSLAM_stereo.open(strORBSLAM_stereoFile.c_str());
    std::vector<Eigen::Vector4d> ORB_stereopath;
    while(!fORBSLAM_stereo.eof())
    {
        double row[12];
        fORBSLAM_stereo >> row[0] >> row[1] >> row[2] >> row[3] >> row[4] >> row[5]
                        >> row[6] >> row[7] >> row[8] >> row[9] >> row[10] >> row[11] ;
        Eigen::Vector4d position=extrinsicMatrix_.inverse()*(Eigen::Vector4d(row[3],row[7],row[11],1));
        ORB_stereopath.emplace_back(position);
    }

    for(int i=0;i<ORB_stereopath.size();i++){
        geometry_msgs::PoseStamped this_pose_stamped;
        this_pose_stamped.pose.position.x = ORB_stereopath.at(i).x();
        this_pose_stamped.pose.position.y = ORB_stereopath.at(i).y();
        this_pose_stamped.pose.position.z = ORB_stereopath.at(i).z();
        this_pose_stamped.header.stamp=ros::Time::now();
        this_pose_stamped.header.frame_id="fusion_slam";
        ORBSLAMstereopath_.poses.emplace_back(this_pose_stamped);
    }
    ORBSLAMstereopath_.header.stamp = ros::Time::now();
    ORBSLAMstereopath_.header.frame_id = "fusion_slam";

    ifstream fORBSLAM_mono;
    string strORBSLAM_monoFile = cfg["File.stereo_noloop"];
    fORBSLAM_mono.open(strORBSLAM_monoFile.c_str());
    std::vector<Eigen::Vector4d> ORB_monopath;
    while(!fORBSLAM_mono.eof())
    {
//            double row[8];
//            fORBSLAM_mono >> row[0] >> row[1] >> row[2] >> row[3] >> row[4] >> row[5]
//                >> row[6] >> row[7] ;
//            Eigen::Vector4d position=extrinsicMatrix_.inverse()*(Eigen::Vector4d(row[1],row[2],row[3],1));
      double row[12];
      fORBSLAM_mono >> row[0] >> row[1] >> row[2] >> row[3] >> row[4] >> row[5]
                    >> row[6] >> row[7] >> row[8] >> row[9] >> row[10] >> row[11] ;
      Eigen::Vector4d position=extrinsicMatrix_.inverse()*(Eigen::Vector4d(row[3],row[7],row[11],1));
      ORB_monopath.emplace_back(position);
    }
    for(int i=0;i<ORB_monopath.size();i++){
        geometry_msgs::PoseStamped this_pose_stamped;
        this_pose_stamped.pose.position.x = ORB_monopath.at(i).x();
        this_pose_stamped.pose.position.y = ORB_monopath.at(i).y();
        this_pose_stamped.pose.position.z = ORB_monopath.at(i).z();
        this_pose_stamped.header.stamp=ros::Time::now();
        this_pose_stamped.header.frame_id="fusion_slam";
        ORBSLAMmonopath_.poses.emplace_back(this_pose_stamped);
    }
    ORBSLAMmonopath_.header.stamp = ros::Time::now();
    ORBSLAMmonopath_.header.frame_id = "fusion_slam";

}


void LT_SLAM::FusionSystem::callback(const sensor_msgs::PointCloud2::ConstPtr &PointCloudMsg){

    lidar_inputPtr_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*PointCloudMsg, *lidar_inputPtr_);

    std::string image_name = ImgFilePath_ + GetFrameStr(frameNum_) + ".png";
    img_gray_ = cv::imread(image_name);

    std::cout << std::endl << std::endl
              << "************************************************************" << std::endl
              << "[FusionSLAM]::帧数:"<< frameNum_ << std::endl
              << "[FusionSLAM]::激光点云数量:" << lidar_inputPtr_->size() << std::endl
              << "-----------------------------前端---------------------------" << std::endl;

    TicToc slam_timer;
    mtx_.lock();
    Tcw_ = SLAM_.TrackFusion(img_gray_, lidar_inputPtr_, vTimestamps_.at(frameNum_));
    frameNum_++;
    mtx_.unlock();

    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << "[FusionSLAM]::SLAM耗时:" << slam_timer.toc() << " ms." << std::endl;

    if(frameNum_==vTimestamps_.size())
        SLAM_.SaveTrajectoryKITTI("FusionSLAM.txt");
}


void LT_SLAM::FusionSystem::Visualization(){
  bool depthline_visual=true;
  bool if_marker_initialization=true;
  while(ros::ok()){

      if(frameNum_>view_framenum_){
        if(view_framenum_==0){// 每次重启程序的时候marker不会重置
          for(int i=0;i<100;i++){
            visualization_msgs::Marker keyframe;
            keyframe.type = visualization_msgs::Marker::CUBE;
            keyframe.header.frame_id = "fusion_slam";
            keyframe.header.stamp = ros::Time::now();
            keyframe.action = visualization_msgs::Marker::ADD;
            keyframe.id = KeyFrameVisual_.markers.size();      // ID counting from 0
            keyframe.lifetime = ros::Duration();
            KeyFrameVisual_.markers.emplace_back(keyframe);
          }
          KFmarker_pub_.publish(KeyFrameVisual_);
        }
          TicToc visual_timer;

          mtx_.lock();
          view_framenum_=frameNum_;
          cv::Mat GrayImg=img_gray_.clone();
          pcl::PointCloud<pcl::PointXYZI> lidar_input=*lidar_inputPtr_;
          pcl::PointCloud<pcl::PointXYZRGB> lidar_color=*lidar_colorPtr_;
          SLAMresult result=SLAM_.getSLAMresult();
          mtx_.unlock();

          cv::Mat Depthimg=result.Depthimg_;
          cv::Mat Range_image=result.range_img_visual_;

          KeyFrameVisual_.markers.clear();
          SLAMpath_.poses.clear();

          Eigen::Matrix4d cur_worldpose(Eigen::Matrix4d::Identity());

          cv::Mat Two = result.vpKFs_[0]->GetPoseInverse();

          long unsigned int referKFid=0;
          // Because of loop detection, trajectory display is related to reference keyframe
          list<ORB_SLAM2::KeyFrame*>::const_iterator lRit = result.lRKFs_.begin();
          for(list<cv::Mat>::const_iterator lit=result.lRPs_.begin(), lend=result.lRPs_.end();
              lit!=lend; lit++, lRit++)
          {
              ORB_SLAM2::KeyFrame* pKF = *lRit;

              cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

              while(pKF->isBad())
              {
                  Trw = Trw*pKF->mTcp;
                  pKF = pKF->GetParent();
              }

              Trw = Trw*pKF->GetPose()*Two;

              cv::Mat Tcw = (*lit)*Trw;
              cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
              cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

              Eigen::Matrix4d KF_cameraPose;
              KF_cameraPose << Rwc.at<float>(0,0),Rwc.at<float>(0,1),Rwc.at<float>(0,2),twc.at<float>(0),
                               Rwc.at<float>(1,0),Rwc.at<float>(1,1),Rwc.at<float>(1,2),twc.at<float>(1),
                               Rwc.at<float>(2,0),Rwc.at<float>(2,1),Rwc.at<float>(2,2),twc.at<float>(2),
                               0,0,0,1;

              Eigen::Matrix4d KFworldpose=extrinsicMatrix_.inverse()*(KF_cameraPose*extrinsicMatrix_);

              geometry_msgs::PoseStamped this_pose_stamped;
              this_pose_stamped.pose.position.x = KFworldpose(0,3);
              this_pose_stamped.pose.position.y = KFworldpose(1,3);
              this_pose_stamped.pose.position.z = KFworldpose(2,3);
              this_pose_stamped.header.stamp=ros::Time::now();
              this_pose_stamped.header.frame_id="fusion_slam";
              SLAMpath_.poses.emplace_back(this_pose_stamped);

              // add keyframe
              if( ((*lRit)->mnId)+1>referKFid ){
                  referKFid=((*lRit)->mnId)+1;
                  Eigen::Matrix3d rotated_matrix=KFworldpose.block(0,0,3,3);
                  Eigen::Quaterniond q=Eigen::Quaterniond(rotated_matrix);
                  q.normalize();
                  tf::Quaternion tfq(q.x(),q.y(),q.z(),q.w());
                  geometry_msgs::Quaternion line_orientation;
                  tf::quaternionTFToMsg( tfq, line_orientation );
                  visualization_msgs::Marker keyframe;
                  keyframe.type = visualization_msgs::Marker::CUBE;
                  keyframe.header.frame_id = "fusion_slam";
                  keyframe.header.stamp = ros::Time::now();
                  keyframe.action = visualization_msgs::Marker::ADD;
                  keyframe.pose.orientation=line_orientation;
                  keyframe.scale.x = 0.2;
                  keyframe.scale.y = 2;
                  keyframe.scale.z = 1.5;
                  keyframe.pose.position = this_pose_stamped.pose.position;
                  keyframe.id = KeyFrameVisual_.markers.size();      // ID counting from 0
                  if( !result.localKFid_.count( (*lRit)->mnId ) )
                    keyframe.color.r = 1.0f;
                  else
                    keyframe.color.g = 1.0f;
//                  keyframe.color.b = 1.0f;
                  keyframe.color.a = 0.8f;
//                  keyframe.lifetime = ros::Duration();
                  KeyFrameVisual_.markers.emplace_back(keyframe);
              }

              //current frame
              if( std::distance(lit,lend)==1 ){
                  cur_worldpose=KFworldpose;
                  tf::Transform transform;
                  Eigen::Matrix3d rotated_matrix=KFworldpose.block(0,0,3,3);
                  Eigen::Quaterniond q=Eigen::Quaterniond(rotated_matrix);
                  q.normalize();
                  tf::Quaternion tfq(q.x(),q.y(),q.z(),q.w());
                  transform.setOrigin(tf::Vector3(cur_worldpose(0,3),cur_worldpose(1,3),cur_worldpose(2,3)));
                  transform.setRotation(tfq);
                  tf_br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/fusion_slam", "/now_position"));
              }
          }

          pcl::PointCloud<pcl::PointXYZ> depthline_pc;

          int count=0;
          for(cv::KeyPoint kp:result.TrackedKeyPoints_){
              if(result.TrackedKeyPoints_.size()==result.KP_viewer_.size()&&result.KP_viewer_.at(count)){
                  if(kp.class_id==0){
                      cv::Point2f pt1,pt2;
                      pt1.x=kp.pt.x-r;
                      pt1.y=kp.pt.y-r;
                      pt2.x=kp.pt.x+r;
                      pt2.y=kp.pt.y+r;
                      cv::rectangle(GrayImg,pt1,pt2,cv::Scalar(0,0,255));
                      cv::circle(GrayImg,kp.pt,2,cv::Scalar(0,0,255),-1);
                  }else{
                      cv::Point2f pt3,pt4;
                      pt3.x=kp.pt.x-half_patch_width;
                      pt3.y=kp.pt.y-half_patch_height;
                      pt4.x=kp.pt.x+half_patch_width;
                      pt4.y=kp.pt.y+half_patch_height;

                      pcl::PointXYZRGB color_p;

                      if(depthline_visual){
                          Eigen::Vector3d normcoor( (kp.pt.x - intrinsicMatrix_(0, 2)) / intrinsicMatrix_(0, 0),
                                                    (kp.pt.y - intrinsicMatrix_(1, 2)) / intrinsicMatrix_(1, 1),
                                                    1 );
                          Eigen::Vector4d invW=Eigen::Vector4d::Zero();
                          while(invW.x()<result.KP_depth_.at(count)){

                              invW=extrinsicMatrix_.inverse()*Eigen::Vector4d(normcoor.x(),normcoor.y(),normcoor.z(),1);
                              pcl::PointXYZ depthline_p;
                              depthline_p._PointXYZ::x=invW.x();
                              depthline_p._PointXYZ::y=invW.y();
                              depthline_p._PointXYZ::z=invW.z();
                              depthline_pc.push_back(depthline_p);
                              double new_z=normcoor.z()+0.05;
                              double new_scale=new_z/normcoor.z();
                              normcoor*=new_scale;
                          }

                          color_p._PointXYZRGB::x=invW.x();
                          color_p._PointXYZRGB::y=invW.y();
                          color_p._PointXYZRGB::z=invW.z();
                      }

                      if(kp.class_id==1){
                          color_p._PointXYZRGB::b=0.0;
                          color_p._PointXYZRGB::g=255.0;
                          color_p._PointXYZRGB::r=0.0;
                          cv::rectangle(GrayImg,pt3,pt4,cv::Scalar(0,255,0));
                          cv::circle(GrayImg,kp.pt,2,cv::Scalar(0,255,0),-1);
                      }else if(kp.class_id==2){
                          color_p._PointXYZRGB::b=255.0;
                          color_p._PointXYZRGB::g=0.0;
                          color_p._PointXYZRGB::r=255.0;
                          cv::rectangle(GrayImg,pt3,pt4,cv::Scalar(0,255,255));
                          cv::circle(GrayImg,kp.pt,2,cv::Scalar(0,255,0),-1);
                      }
                      if(depthline_visual)    lidar_color.push_back(color_p);

                  }
              }
              count++;
          }


          pcl::PointCloud<pcl::PointXYZ> map_pc;
          for(ORB_SLAM2::MapPoint* mp:result.map_){
              if(mp){
                  cv::Mat camera_worldP=mp->GetWorldPos();
                  Eigen::Matrix<double,3,1> camera_WP=ORB_SLAM2::Converter::toVector3d(camera_worldP);
                  Eigen::Vector4d lidar_WP=extrinsicMatrix_.inverse()*
                      (Eigen::Vector4d(camera_WP.x(),camera_WP.y(),camera_WP.z(),1));
                  pcl::PointXYZ map_p;
                  map_p._PointXYZ::x=lidar_WP.x();
                  map_p._PointXYZ::y=lidar_WP.y();
                  map_p._PointXYZ::z=lidar_WP.z();
                  map_pc.push_back(map_p);
              }
          }

          pcl::PointCloud<pcl::PointXYZ> localmap_pc;
          for(ORB_SLAM2::MapPoint* localmp:result.localmap_){
              if(localmp){
                  cv::Mat camera_worldP=localmp->GetWorldPos();
                  Eigen::Matrix<double,3,1> camera_WP=ORB_SLAM2::Converter::toVector3d(camera_worldP);
                  Eigen::Vector4d lidar_WP=extrinsicMatrix_.inverse()*
                      (Eigen::Vector4d(camera_WP.x(),camera_WP.y(),camera_WP.z(),1));
                  pcl::PointXYZ localmap_p;
                  localmap_p._PointXYZ::x=lidar_WP.x();
                  localmap_p._PointXYZ::y=lidar_WP.y();
                  localmap_p._PointXYZ::z=lidar_WP.z();
                  localmap_pc.push_back(localmap_p);
              }
          }

          Eigen::Matrix4d extrinsic_inv=extrinsicMatrix_.inverse();
          Eigen::Matrix4d pose_lidarfeature=cur_worldpose*extrinsic_inv;

          pcl::transformPointCloud(lidar_input, lidar_input, cur_worldpose);
          pcl::transformPointCloud(lidar_color, lidar_color, cur_worldpose);
          pcl::transformPointCloud(depthline_pc, depthline_pc, cur_worldpose);
          pcl::transformPointCloud(result.corner_points_sharp_, result.corner_points_sharp_, pose_lidarfeature);
          pcl::transformPointCloud(result.surface_points_flat_, result.surface_points_flat_, pose_lidarfeature);
          pcl::transformPointCloud(result.corner_points_less_sharp_, result.corner_points_less_sharp_, pose_lidarfeature);
          pcl::transformPointCloud(result.surface_points_less_flat_, result.surface_points_less_flat_, pose_lidarfeature);

          pcl::transformPointCloud(result.lidar_local_map_, result.lidar_local_map_, extrinsic_inv);


          Depthimg.convertTo(Depthimg,CV_8UC3);
          cv::normalize(Depthimg,Depthimg,255.0,0.0,cv::NORM_MINMAX);//归一到0~255之间
          cv::Mat im_color;
          cv::applyColorMap(Depthimg,im_color,cv::COLORMAP_JET);
          cv_bridge::CvImage depthimg_bridge;
          sensor_msgs::Image depthimg_msg;
          std_msgs::Header depthimg_header;
          depthimg_header.stamp = ros::Time::now();
          depthimg_header.frame_id="fusion_slam";
          depthimg_bridge = cv_bridge::CvImage(depthimg_header, sensor_msgs::image_encodings::BGR8, im_color);
          depthimg_bridge.toImageMsg(depthimg_msg);

          Range_image.convertTo(Range_image,CV_8UC3);
          cv::normalize(Range_image,Range_image,255.0,0.0,cv::NORM_MINMAX);//归一到0~255之间
          cv::Mat Range_image_color;
          cv::applyColorMap(Range_image,Range_image_color,cv::COLORMAP_JET);
          cv_bridge::CvImage rangeimg_bridge;
          sensor_msgs::Image rangeimg_msg;
          std_msgs::Header rangeimg_header;
          rangeimg_header.stamp = ros::Time::now();
          rangeimg_header.frame_id="fusion_slam";
          rangeimg_bridge = cv_bridge::CvImage(rangeimg_header, sensor_msgs::image_encodings::BGR8, Range_image_color);
          rangeimg_bridge.toImageMsg(rangeimg_msg);

          cv_bridge::CvImage img_bridge;
          sensor_msgs::Image img_msg;
          std_msgs::Header header;
          header.stamp = ros::Time::now();
          header.frame_id="fusion_slam";
          img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, GrayImg);
          img_bridge.toImageMsg(img_msg);

          sensor_msgs::PointCloud2 rotated_pcros;
          pcl::toROSMsg(lidar_input,rotated_pcros);
          rotated_pcros.header.frame_id="fusion_slam";

          sensor_msgs::PointCloud2 color_pcros;
          pcl::toROSMsg(lidar_color,color_pcros);
          color_pcros.header.frame_id="fusion_slam";

          sensor_msgs::PointCloud2 depthline_pcros;
          pcl::toROSMsg(depthline_pc,depthline_pcros);
          depthline_pcros.header.frame_id="fusion_slam";

          sensor_msgs::PointCloud2 map_pcros;
          pcl::toROSMsg(map_pc,map_pcros);
          map_pcros.header.frame_id="fusion_slam";

          sensor_msgs::PointCloud2 localmap_pcros;
          pcl::toROSMsg(localmap_pc,localmap_pcros);
          localmap_pcros.header.frame_id="fusion_slam";

          SLAMpath_.header.stamp = ros::Time::now();
          SLAMpath_.header.frame_id = "fusion_slam";

          sensor_msgs::PointCloud2 sharp_pcros;
          pcl::toROSMsg(result.corner_points_sharp_,sharp_pcros);
          sharp_pcros.header.frame_id="fusion_slam";

          sensor_msgs::PointCloud2 flat_pcros;
          pcl::toROSMsg(result.surface_points_flat_,flat_pcros);
          flat_pcros.header.frame_id="fusion_slam";

          sensor_msgs::PointCloud2 less_sharp_pcros;
          pcl::toROSMsg(result.corner_points_less_sharp_,less_sharp_pcros);
          less_sharp_pcros.header.frame_id="fusion_slam";

          sensor_msgs::PointCloud2 less_flat_pcros;
          pcl::toROSMsg(result.surface_points_less_flat_,less_flat_pcros);
          less_flat_pcros.header.frame_id="fusion_slam";

          sensor_msgs::PointCloud2 lidar_local_map_pcros;
          pcl::toROSMsg(result.lidar_local_map_,lidar_local_map_pcros);
          lidar_local_map_pcros.header.frame_id="fusion_slam";

          depthimage_pub_.publish(depthimg_msg);
          image_pub_.publish(img_msg);
          rangeimage_pub_.publish(rangeimg_msg);
          rotated_pc_pub_.publish(rotated_pcros);
          color_pc_pub_.publish(color_pcros);
          depthline_pub_.publish(depthline_pcros);
          map_pub_.publish(map_pcros);
          localmap_pub_.publish(localmap_pcros);

          sharp_cloud_pub_.publish(sharp_pcros);
          flat_cloud_pub_.publish(flat_pcros);
          less_sharp_cloud_pub_.publish(less_sharp_pcros);
          less_flat_cloud_pub_.publish(less_flat_pcros);
          lidar_local_map_pub_.publish(lidar_local_map_pcros);

          SLAMpath_pub_.publish(SLAMpath_);
          KFmarker_pub_.publish(KeyFrameVisual_);
          groundtruth_pub_.publish(GTpath_);
          ORBSLAMstereopath_pub_.publish(ORBSLAMstereopath_);
          ORBSLAMmonopath_pub_.publish(ORBSLAMmonopath_);

          std::cout << "[FusionSLAM]::可视化耗时:" << visual_timer.toc() << " ms." << std::endl;
      }
      std::chrono::milliseconds dura(2);
      std::this_thread::sleep_for(dura);
  }
}

//void LT_SLAM::FusionSystem::DataConversion(const sensor_msgs::PointCloud2::ConstPtr &PointCloudMsg){

//    std::lock_guard<mutex> lock(mtx_);

//    lidar_inputPtr_.reset(new pcl::PointCloud<pcl::PointXYZI>);
//    lidar_colorPtr_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::fromROSMsg(*PointCloudMsg, *lidar_inputPtr_);

//    std::string image_name = ImgFilePath_ + GetFrameStr(frameNum_) + ".png";

//    curFrame_.GrayImg = cv::imread(image_name);
//    curFrame_.Depthimg = cv::Mat(curFrame_.GrayImg.rows, curFrame_.GrayImg.cols, CV_64F, cv::Scalar::all(0));

//    min_v_=10000;
//    for(int i = 0; i < lidar_inputPtr_->size(); i++){

//        if(lidar_inputPtr_->points[i].x<0)  continue;

//        cv::Point2i pixel;
//        Eigen::Vector4d P_lidar(lidar_inputPtr_->points[i].x,
//                                lidar_inputPtr_->points[i].y,
//                                lidar_inputPtr_->points[i].z,
//                                1);

//        Eigen::Vector3i P_uv = TransformProject(P_lidar);

//        if(P_uv[0] >= 0 && P_uv[1] >= 0 && P_uv[0]<=curFrame_.GrayImg.cols-1 && P_uv[1]<=curFrame_.GrayImg.rows-1){

//            pixel.x = P_uv[0];
//            pixel.y = P_uv[1];

//            if(pixel.y<min_v_) min_v_=pixel.y;

//            if(lidar_inputPtr_->points[i].x>1){
//                pcl::PointXYZRGB color_p;
//                color_p._PointXYZRGB::x=lidar_inputPtr_->points[i].x;
//                color_p._PointXYZRGB::y=lidar_inputPtr_->points[i].y;
//                color_p._PointXYZRGB::z=lidar_inputPtr_->points[i].z;
//                color_p._PointXYZRGB::b=curFrame_.GrayImg.at<cv::Vec3b>(P_uv[1],P_uv[0])[0];
//                color_p._PointXYZRGB::g=curFrame_.GrayImg.at<cv::Vec3b>(P_uv[1],P_uv[0])[1];
//                color_p._PointXYZRGB::r=curFrame_.GrayImg.at<cv::Vec3b>(P_uv[1],P_uv[0])[2];
//                lidar_colorPtr_->push_back(color_p);

//                curFrame_.Depthimg.at<double>(pixel.y,pixel.x)=(extrinsicMatrix_*P_lidar).z();
//            }
//        }
//    }

//}
