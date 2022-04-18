#define PCL_NO_PRECOMPILE
#include "algorithm/lidarOdom.h"
#include <pcl/registration/gicp.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <cv_bridge/cv_bridge.h>

namespace art {
void OdomConfig::SetupParam(ros::NodeHandle &nh) {
    nh.param("even", even,false);
    nh.param("sensor_type", sensor_type, 32);
    nh.param("deg_diff", deg_diff, 0.2);
    nh.param("scan_period", scan_period, 0.1);
    nh.param("lower_bound", lower_bound, -25.0);
    nh.param("upper_bound", upper_bound, 15.0);
    nh.param("using_lower_ring_num", using_lower_ring_num, 1);
    nh.param("using_upper_ring_num", using_upper_ring_num, 32);
    nh.param("num_scan_subregions", num_scan_subregions, 8);
    nh.param("num_curvature_regions_corner", num_curvature_regions_corner, 1);
    nh.param("num_curvature_regions_flat", num_curvature_regions_flat, 5);
    nh.param("num_feature_regions", num_feature_regions, 5);
    nh.param("ground_z_bound", ground_z_bound, -1.0);
    nh.param("surf_curv_th", surf_curv_th, 0.1);
    nh.param("sharp_curv_th", sharp_curv_th, 1.0);
    nh.param("max_corner_sharp", max_corner_sharp, 4);
    nh.param("max_corner_less_sharp", max_corner_less_sharp, 20);
    nh.param("max_surf_flat", max_surf_flat, 2);
    nh.param("max_surf_less_flat", max_surf_less_flat, 4);
    nh.param("max_sq_dis", max_sq_dis, 1.0);
    nh.param("flat_extract_num_x_trans", flat_extract_num_x_trans, 50);
    nh.param("flat_extract_num_y_trans", flat_extract_num_y_trans, 50);
    nh.param("flat_extract_num_z_trans", flat_extract_num_z_trans, 50);
    nh.param("flat_extract_num_x_rot", flat_extract_num_x_rot, 50);
    nh.param("flat_extract_num_y_rot", flat_extract_num_y_rot, 50);
    nh.param("flat_extract_num_z_rot", flat_extract_num_z_rot, 50);
    nh.param("lower_ring_num_sharp_point", lower_ring_num_sharp_point, 1);
    nh.param("upper_ring_num_sharp_point", upper_ring_num_sharp_point, 32);
    nh.param("lower_ring_num_z_trans", lower_ring_num_z_trans, 1);
    nh.param("upper_ring_num_z_trans", upper_ring_num_z_trans, 20);
    nh.param("lower_ring_num_x_rot", lower_ring_num_x_rot, 1);
    nh.param("upper_ring_num_x_rot", upper_ring_num_x_rot, 20);
    nh.param("lower_ring_num_y_rot", lower_ring_num_y_rot, 1);
    nh.param("upper_ring_num_y_rot", upper_ring_num_y_rot, 20);
    nh.param("lower_ring_num_z_rot_xy_trans", lower_ring_num_z_rot_xy_trans, 10);
    nh.param("upper_ring_num_z_rot_xy_trans", upper_ring_num_z_rot_xy_trans, 32);
    nh.param("less_flat_filter_size", less_flat_filter_size, 0.2);
    nh.param("using_corner_point_vector", using_corner_point_vector, true);
    nh.param("using_surf_point_normal", using_surf_point_normal, true);
    nh.param("using_sharp_point", using_sharp_point, true);
    nh.param("using_flat_point", using_flat_point, true);
    nh.param("distance_sq_threshold", distance_sq_threshold, 0.2);
}

void OdomConfig::setParam(cv::FileStorage &cfg){
  int using_corner_point_vector_i = cfg["using_corner_point_vector"];
  using_corner_point_vector=using_corner_point_vector_i;
  int using_surf_point_normal_i = cfg["using_surf_point_normal"];
  using_surf_point_normal=using_surf_point_normal_i;
  int using_sharp_point_i = cfg["using_sharp_point"];
  using_sharp_point=using_sharp_point_i;
  int using_flat_point_i = cfg["using_flat_point"];
  using_flat_point=using_flat_point_i;
  sensor_type = cfg["sensor_type"];
  deg_diff = cfg["deg_diff"];
  scan_period = cfg["scan_period"];
  lower_bound = cfg["lower_bound"];
  upper_bound = cfg["upper_bound"];
  ground_z_bound = cfg["ground_z_bound"];                              // 地面高度
  using_lower_ring_num = cfg["using_lower_ring_num"];
  using_upper_ring_num = cfg["using_upper_ring_num"];
  lower_ring_num_sharp_point = cfg["lower_ring_num_sharp_point"];      // 角点范围
  upper_ring_num_sharp_point = cfg["upper_ring_num_sharp_point"];      // 角点范围
  lower_ring_num_z_trans = cfg["lower_ring_num_z_trans"];              // 估计z平移的平面点范围
  upper_ring_num_z_trans = cfg["upper_ring_num_z_trans"];              // 估计z平移的平面点范围
  lower_ring_num_x_rot = cfg["lower_ring_num_x_rot"];                  // 估计x旋转的平面点范围
  upper_ring_num_x_rot = cfg["upper_ring_num_x_rot"];                  // 估计x旋转的平面点范围
  lower_ring_num_y_rot = cfg["lower_ring_num_y_rot"];                  // 估计y旋转的平面点范围
  upper_ring_num_y_rot = cfg["upper_ring_num_y_rot"];                  // 估计y旋转的平面点范围
  lower_ring_num_z_rot_xy_trans = cfg["lower_ring_num_z_rot_xy_trans"];// 估计z旋转和xy平移的平面点范围
  upper_ring_num_z_rot_xy_trans = cfg["upper_ring_num_z_rot_xy_trans"];// 估计z旋转和xy平移的平面点范围
  num_scan_subregions = cfg["num_scan_subregions"];                    // 一线分为多少个子区域
  num_curvature_regions_corner = cfg["num_curvature_regions_corner"];  // 算角点时候的邻域
  num_curvature_regions_flat = cfg["num_curvature_regions_flat"];      // 算平面点时候的邻域
  num_feature_regions = cfg["num_feature_regions"];
  surf_curv_th = cfg["surf_curv_th"];                                  // 平面点曲率阈值  越小越平
  sharp_curv_th = cfg["sharp_curv_th"];                                // 角点曲率阈值  越大越好
  max_corner_sharp = cfg["max_corner_sharp"];                          // 前max_corner_sharp个曲率大的点被设置为角点
  max_corner_less_sharp = cfg["max_corner_less_sharp"];                // 前max_corner_less_sharp个曲率大的点被设置为弱角点
  max_surf_flat = cfg["max_surf_flat"];
  max_surf_less_flat = cfg["max_surf_less_flat"];
  max_sq_dis = cfg["max_sq_dis"];                                      // 提取平面点时的邻域大小
  flat_extract_num_x_trans = cfg["flat_extract_num_x_trans"];          // 使用这部分点的数量
  flat_extract_num_y_trans = cfg["flat_extract_num_y_trans"];
  flat_extract_num_z_trans = cfg["flat_extract_num_z_trans"];
  flat_extract_num_x_rot = cfg["flat_extract_num_x_rot"];
  flat_extract_num_y_rot = cfg["flat_extract_num_y_rot"];
  flat_extract_num_z_rot = cfg["flat_extract_num_z_rot"];
  less_flat_filter_size = cfg["less_flat_filter_size"];
  distance_sq_threshold = cfg["distance_sq_threshold"];
}


Odom::Odom(const OdomConfig &config, const ros::NodeHandle &nh) : config_(config), nh_(nh) {
  pub_processed_cloud_  = nh_.advertise<sensor_msgs::PointCloud2>("/ring_cloud", 100);
  pub_sharp_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/sharp_cloud", 100);
  pub_flat_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/flat_cloud", 100);
  pub_less_sharp_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/less_sharp_cloud", 100);
  pub_less_flat_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/less_flat_cloud", 100);
  pub_localmap_ = nh_.advertise<sensor_msgs::PointCloud2>("/local_map", 100);
  pub_rangeimage_ = nh_.advertise<sensor_msgs::Image>("/lidardepth_image", 1);

  time_factor_ = 1/config_.scan_period;
  factor_ = ((config_.sensor_type -1) / (config_.upper_bound - config_.lower_bound));

  row_num_ = config_.sensor_type;
  col_num_ = 360.0/config_.deg_diff;

  range_img_visual_ = cv::Mat(row_num_, col_num_, CV_64F, cv::Scalar::all(0));

  laser_scans_.clear();
  for (int i = 0; i < row_num_; ++i) {
    PointIRTCloudPtr scan(new PointIRTCloud());
    laser_scans_.push_back(scan);
  }

  range_image_.clear();
  for (int i = 0; i < row_num_; ++i) {
    std::vector<ImageElement> image_row(col_num_);
    range_image_.push_back(image_row);
  }

  map_cloud_ptr_ = PointIRTCloud().makeShared();
  local_map_cloud_ptr_ = PointIRTCloud().makeShared();

  kdtree_map_ = pcl::KdTreeFLANN<PointIRT>().makeShared();
  kdtree_local_map_ = pcl::KdTreeFLANN<PointIRT>().makeShared();

  system_inited_ = false;
  transf_last_curr_.setIdentity();
  estimate_pose_.setIdentity();
}


void Odom::pubFeaturePoint(){
  PointIRTCloud transformed_cloud;
  pcl::transformPointCloud(cloud_in_rings_, transformed_cloud, estimate_pose_);
  sensor_msgs::PointCloud2 processed_cloud;
  pcl::toROSMsg(transformed_cloud, processed_cloud);
  processed_cloud.header.stamp = ros::Time::now();
  processed_cloud.header.frame_id = "/velodyne";
  pub_processed_cloud_.publish(processed_cloud);

  pcl::transformPointCloud(corner_points_sharp_, transformed_cloud, estimate_pose_);
  sensor_msgs::PointCloud2 sharp_cloud;
  pcl::toROSMsg(transformed_cloud, sharp_cloud);
  sharp_cloud.header.stamp = ros::Time::now();
  sharp_cloud.header.frame_id = "/velodyne";
  pub_sharp_cloud_.publish(sharp_cloud);

  pcl::transformPointCloud(surface_points_flat_, transformed_cloud, estimate_pose_);
  sensor_msgs::PointCloud2 flat_cloud;
  pcl::toROSMsg(transformed_cloud, flat_cloud);
  flat_cloud.header.stamp = ros::Time::now();
  flat_cloud.header.frame_id = "/velodyne";
  pub_flat_cloud_.publish(flat_cloud);

  pcl::transformPointCloud(corner_points_less_sharp_, transformed_cloud, estimate_pose_);
  sensor_msgs::PointCloud2 less_sharp_cloud;
  pcl::toROSMsg(transformed_cloud, less_sharp_cloud);
  less_sharp_cloud.header.stamp = ros::Time::now();
  less_sharp_cloud.header.frame_id = "/velodyne";
  pub_less_sharp_cloud_.publish(less_sharp_cloud);

  pcl::transformPointCloud(surface_points_less_flat_, transformed_cloud, estimate_pose_);
  sensor_msgs::PointCloud2 less_flat_cloud;
  pcl::toROSMsg(transformed_cloud, less_flat_cloud);
//  pcl::toROSMsg(mask_points_, less_flat_cloud);
  less_flat_cloud.header.stamp = ros::Time::now();
  less_flat_cloud.header.frame_id = "/velodyne";
  pub_less_flat_cloud_.publish(less_flat_cloud);

  sensor_msgs::PointCloud2 local_map;
  pcl::toROSMsg(*local_map_cloud_ptr_, local_map);
  local_map.header.stamp = ros::Time::now();
  local_map.header.frame_id = "/velodyne";
  pub_localmap_.publish(local_map);

  cv::Mat range_img_visual=range_img_visual_.clone();
  range_img_visual.convertTo(range_img_visual,CV_8UC3);
  cv::normalize(range_img_visual,range_img_visual,255.0,0.0,cv::NORM_MINMAX);//归一到0~255之间
  cv::Mat im_color;
  cv::applyColorMap(range_img_visual,im_color,cv::COLORMAP_JET);
  cv_bridge::CvImage depthimg_bridge;
  sensor_msgs::Image depthimg_msg;
  std_msgs::Header depthimg_header;
  depthimg_header.stamp = ros::Time::now();
  depthimg_header.frame_id="velodyne";
  depthimg_bridge = cv_bridge::CvImage(depthimg_header, sensor_msgs::image_encodings::BGR8, im_color);
  depthimg_bridge.toImageMsg(depthimg_msg);
  pub_rangeimage_.publish(depthimg_msg);
}


void Odom::GetFeaturePoints(const PointIRTCloud &all_cloud_in, PointIRTCloud &feature_cloud) {
  Reset();
  PointToImage(all_cloud_in);
  ExtractFeaturePoints();
  pubFeaturePoint();
  feature_cloud += corner_points_less_sharp_;
  feature_cloud += surface_points_less_flat_;
}


void Odom::EstimatePoseForSLAM(const PointIRTCloud &all_cloud_in, Eigen::Affine3d &estimate_pose) {
  Reset();
  PointToImage(all_cloud_in);
  ExtractFeaturePoints();
  FrontEndForSLAM();
  estimate_pose = estimate_pose_;
}


void Odom::EstimatePoseForSLAM(const PointIRTCloud &all_cloud_in, Eigen::Affine3d &estimate_pose, Eigen::Affine3d &transf_last_curr) {
  Reset();
  PointToImage(all_cloud_in);
  ExtractFeaturePoints();
  transf_last_curr_ = transf_last_curr;
  FrontEndForSLAM();
  estimate_pose = estimate_pose_;
  pubFeaturePoint();
}


void Odom::EstimatePoseForSLAM(const PointIRTCloud &all_cloud_in, PointIRTCloud &feature_cloud, Eigen::Affine3d &estimate_pose) {
  Reset();
  PointToImage(all_cloud_in);
  ExtractFeaturePoints();
  FrontEndForSLAM();
  feature_cloud += corner_points_less_sharp_;
  feature_cloud += surface_points_less_flat_;
  estimate_pose = estimate_pose_;
}


void Odom::EstimatePoseForMapping(const PointIRTCloud &all_cloud_in,  PointIRTCloud &feature_cloud, Eigen::Affine3d &estimate_pose, Eigen::Affine3d &transf_last_curr) {
  Reset();
  PointToImage(all_cloud_in);
  ExtractFeaturePoints();
  transf_last_curr_ = transf_last_curr;
  estimate_pose_ = estimate_pose;
  FrontEndForMapping();
  feature_cloud += corner_points_less_sharp_;
  feature_cloud += surface_points_less_flat_;
  estimate_pose = estimate_pose_;
  transf_last_curr = transf_last_curr_;
}


void Odom::EstimatePoseForMapping(const std::deque<MapFrame> &local_map_frames, std::deque<MapFrame> &estimate_frames, Eigen::Affine3d &transf_last_curr) {
  system_inited_ = false;
  transf_last_curr_ = transf_last_curr;
  estimate_pose_ = estimate_frames.front().transform;
  Eigen::Affine3d last_frame_pose = estimate_frames.back().transform;

  std::vector<Eigen::Affine3d> frame_poses;
  size_t frame_num = estimate_frames.size();
  for (size_t i = 0; i < frame_num; ++i) {
    Reset();
    PointToImage(estimate_frames[i].all_points);
    ExtractFeaturePoints();
    if (i > 0) {
      FrontEndForMapping(local_map_frames);
    }
    frame_poses.push_back(estimate_pose_);
    estimate_frames[i].feature_points += corner_points_less_sharp_;
    estimate_frames[i].feature_points += surface_points_less_flat_;
  }

  // BackEndForLoop(frame_poses, last_frame_pose);
  BackEndForGNSS(frame_poses, last_frame_pose);
  for (size_t i = 0; i < frame_num; ++i) {
    estimate_frames[i].transform = frame_poses[i];
  }
}


void Odom::EstimatePoseForLocalization(const PointIRTCloud &all_cloud_in, Eigen::Affine3d &estimate_pose, Eigen::Affine3d &transf_last_curr) {
  Reset();
  PointToImage(all_cloud_in);
  ExtractFeaturePoints();
  transf_last_curr_ = transf_last_curr;
  estimate_pose_ = estimate_pose;
  FrontEndForLocalization();
  estimate_pose = estimate_pose_;
}


void Odom::Reset() {
  // clear cloud buffers
  cloud_in_rings_.clear();
  corner_points_sharp_.clear();
  corner_points_less_sharp_.clear();
  surface_points_flat_.clear();
  surface_points_less_flat_.clear();
  surface_points_less_flat_index_.clear();
  surface_points_flat_z_trans_.clear();
  surface_points_flat_z_rot_xy_trans_.clear();
  surface_points_flat_x_rot_.clear();
  surface_points_flat_y_rot_.clear();

  corner_points_vector_.clear();
  surface_points_flat_normal_.clear();
  surface_points_less_flat_normal_.clear();
  mask_points_.clear();

  laser_scans_.clear();
  for (int i = 0; i < row_num_; ++i) {
    PointIRTCloudPtr scan(new PointIRTCloud());
    laser_scans_.push_back(scan);
  }

  // clear scan indices vector
  scan_ranges_.clear();

  // clear image indices vector
  image_index_.clear();
  image_index_.resize(row_num_);

  range_image_.clear();
  for (int i = 0; i < row_num_; ++i) {
    std::vector<ImageElement> image_row(col_num_);
    range_image_.push_back(image_row);
  }

  less_sharp_.clear();
  sharp_.clear();

  range_img_visual_.setTo(0);
}


void Odom::PointToImage(const PointIRTCloud &all_cloud_in) {
  auto &points = all_cloud_in.points;
  size_t all_cloud_size = points.size();
  float startOri = -atan2(points[0].y, points[0].x) + M_PI;
  float endOri = -atan2(points[all_cloud_size - 1].y,
                        points[all_cloud_size - 1].x) + 3 * M_PI;
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

  bool halfPassed = false;
  std::vector<int> index_num(config_.sensor_type, 0);    // 每一行多少点
  int ground_pointnum=0;

  for (size_t i = 0; i < all_cloud_size; ++i) {

    float azi_rad = -atan2(points[i].y, points[i].x);
    if (!halfPassed) {
      if (azi_rad < startOri - M_PI / 2) {
        azi_rad += 2 * M_PI;
      } else if (azi_rad > startOri + M_PI * 3 / 2) {
        azi_rad -= 2 * M_PI;
      }
      if (azi_rad - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      azi_rad += 2 * M_PI;
      if (azi_rad < endOri - M_PI * 3 / 2) {
        azi_rad += 2 * M_PI;
      } else if (azi_rad > endOri + M_PI / 2) {
        azi_rad -= 2 * M_PI;
      }
    }

    float azi_rad_rel = azi_rad - startOri;

    int col = std::fmod(int(azi_rad_rel / (2 * M_PI) * col_num_) + col_num_, col_num_);
    int row = points[i].ring;

    if (col < 0 || col >= col_num_ || row < 0 || row >= row_num_) {
      continue;
    }

    // -3=没有占用  -1=地面点   0=非地面点
    if (range_image_[row][col].point_state == -3) {
      if (points[i].z < config_.ground_z_bound) {
        range_image_[row][col].point_state = -1;
        ground_pointnum++;
      } else {
        range_image_[row][col].point_state = 0;
      }
      range_image_[row][col].index = index_num[row];
      laser_scans_[row]->push_back(points[i]);
      ImageIndex index_temp;
      index_temp.row = row;
      index_temp.col = col;
      image_index_[row].push_back(index_temp);
      ++index_num[row];

      range_img_visual_.at<double>(row,col)=CalcPointDistance(points[i]);

    } else {
      //如果已经占用选择距离更近的那个
      int scan_index = range_image_[row][col].index;
      float point_dis_1 = CalcPointDistance(points[i]);
      float point_dis_2 = CalcPointDistance(laser_scans_[row]->points[scan_index]);
      if (point_dis_1 < point_dis_2) {
        if (points[i].z < config_.ground_z_bound) {
          range_image_[row][col].point_state = -1;
          ground_pointnum++;
        } else {
          range_image_[row][col].point_state = 0;
        }
        laser_scans_[row]->points[scan_index] = points[i];
        range_img_visual_.at<double>(row,col)=CalcPointDistance(points[i]);
      }
    }
  }

  size_t cloud_size = 0;
  for (int i = 0; i < row_num_; i++) {
    cloud_in_rings_ += (*laser_scans_[i]);
    IndexRange range(cloud_size, 0);
    cloud_size += (*laser_scans_[i]).size();
    range.second = (cloud_size > 0 ? cloud_size - 1 : 0);
    scan_ranges_.push_back(range);
    // 比如说第一行18个点  第二行19个点  scan_ranges_[0]就是（0.17）  scan_ranges_[2]=（18,18+19）
  }
  std::cout << "[PointToImage]::深度图化点数：" << cloud_size
            << "        地面点数：" << ground_pointnum << std::endl;
}


// 提取角点前的准备，主要是设置这个掩膜，根据同一线点和点的距离关系来设置，设置了掩膜（1）的就不做后面的判断了
void Odom::PrepareRing_corner(const PointIRTCloud &scan) {

  size_t scan_size = scan.size();
  //预处理掩膜
  scan_ring_mask_.resize(scan_size);
  scan_ring_mask_.assign(scan_size, 0);
  // // 记录每个scan的结束index，忽略后n个点，开始和结束处的点云容易产生不闭合的“接缝”，对提取edge feature不利
  for (size_t i = 0 + config_.num_curvature_regions_corner; i + config_.num_curvature_regions_corner < scan_size; ++i) {
    const PointIRT &p_prev = scan[i - 1];
    const PointIRT &p_curr = scan[i];
    const PointIRT &p_next = scan[i + 1];

    float diff_next2 = CalcSquaredDiff(p_curr, p_next);

    // about 30 cm 如果和下一个点的距离超过30cm
    if (diff_next2 > 0.1) {
      float depth = CalcPointDistance(p_curr);
      float depth_next = CalcPointDistance(p_next);

      // 比较深度
      if (depth > depth_next) {
        // to closer point
        float weighted_diff = sqrt(CalcSquaredDiff(p_next, p_curr, depth_next / depth)) / depth_next;
        // relative distance
        if (weighted_diff < 0.1) {
          // 把上num_curvature_regions_corner个点到当前点掩膜置位
          fill_n(&scan_ring_mask_[i - config_.num_curvature_regions_corner], config_.num_curvature_regions_corner + 1, 1);
          continue;
        }
      } else {
        float weighted_diff = sqrt(CalcSquaredDiff(p_curr, p_next, depth / depth_next)) / depth;
        if (weighted_diff < 0.1) {
          // 把下num_curvature_regions_corner个点置位
          fill_n(&scan_ring_mask_[i + 1], config_.num_curvature_regions_corner, 1);
          continue;
        }
      }
    }

    float diff_prev2 = CalcSquaredDiff(p_curr, p_prev);
    float dis2 = CalcSquaredPointDistance(p_curr);

    // for this point -- 1m -- 1.5cm
    if (diff_next2 > 0.0002 * dis2 && diff_prev2 > 0.0002 * dis2) {
      scan_ring_mask_[i - 0] = 1;
    }
  }

//  for(int i=0; i<scan_size; i++){
//    if(scan_ring_mask_[1]==1){
//      mask_points_.push_back(scan[i]);
//    }
//  }

}


void Odom::PrepareRing_flat(const PointIRTCloud &scan) {

  size_t scan_size = scan.size();
  scan_ring_mask_.resize(scan_size);
  scan_ring_mask_.assign(scan_size, 0);
  for (size_t i = 0 + config_.num_curvature_regions_flat; i + config_.num_curvature_regions_flat < scan_size; ++i) {
    const PointIRT &p_prev = scan[i - 1];
    const PointIRT &p_curr = scan[i];
    const PointIRT &p_next = scan[i + 1];

    float diff_next2 = CalcSquaredDiff(p_curr, p_next);

    // about 30 cm
    if (diff_next2 > 0.1) {
      float depth = CalcPointDistance(p_curr);
      float depth_next = CalcPointDistance(p_next);

      if (depth > depth_next) {
        // to closer point
        // 是区分两个点到激光雷达的向量  基本保持一条直线
        float weighted_diff = sqrt(CalcSquaredDiff(p_next, p_curr, depth_next / depth)) / depth_next;
        // relative distance
        if (weighted_diff < 0.1) {
          fill_n(&scan_ring_mask_[i - config_.num_curvature_regions_flat], config_.num_curvature_regions_flat + 1, 1);
          continue;
        }
      } else {
        float weighted_diff = sqrt(CalcSquaredDiff(p_curr, p_next, depth / depth_next)) / depth;
        if (weighted_diff < 0.1) {
          fill_n(&scan_ring_mask_[i + 1], config_.num_curvature_regions_flat, 1);
          continue;
        }
      }
    }

//     float diff_prev2 = CalcSquaredDiff(p_curr, p_prev);
//     float dis2 = CalcSquaredPointDistance(p_curr);

//     // for this point -- 1m -- 1.5cm
//     if (diff_next2 > 0.0002 * dis2 && diff_prev2 > 0.0002 * dis2) {
//       scan_ring_mask_[i - 0] = 1;
//     }
  }
//  for(int i=0; i<scan_size; i++){
//    if(scan_ring_mask_[1]==1){
//      mask_points_.push_back(scan[i]);
//    }
//  }
}


/// 算曲率
void Odom::PrepareSubregion_corner(const PointIRTCloud &scan, const size_t idx_start, const size_t idx_end) {

//  cout << ">>>>>>> " << idx_ring << ", " << idx_start << ", " << idx_end << " <<<<<<<" << endl;
//  const PointIRTCloud &scan = laser_scans_[idx_ring];
  size_t region_size = idx_end - idx_start + 1;
  curvature_idx_pairs_.clear();
  curvature_idx_pairs_.resize(region_size);

  // 算曲率  邻域是左右多少点  LOAM中的公式，那个曲率其实就代表弯曲的模长
  // https://blog.csdn.net/shoufei403/article/details/103664877
  for (size_t i = idx_start, in_region_idx = 0; i <= idx_end; ++i, ++in_region_idx) {

    int num_point_neighbors = 2 * config_.num_curvature_regions_corner;
    float diff_x = -num_point_neighbors * scan[i].x;
    float diff_y = -num_point_neighbors * scan[i].y;
    float diff_z = -num_point_neighbors * scan[i].z;

    for (int j = 1; j <= config_.num_curvature_regions_corner; ++j) {
      diff_x += scan[i + j].x + scan[i - j].x;
      diff_y += scan[i + j].y + scan[i - j].y;
      diff_z += scan[i + j].z + scan[i - j].z;
    }


    float curvature = (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)/num_point_neighbors;
    pair<float, size_t> curvature_idx_(curvature, i);
    curvature_idx_pairs_[in_region_idx] = curvature_idx_;
//    _regionCurvature[regionIdx] = diffX * diffX + diffY * diffY + diffZ * diffZ;
//    _regionSortIndices[regionIdx] = i;
  }

  sort(curvature_idx_pairs_.begin(), curvature_idx_pairs_.end());
/*
  for (const auto &pair : curvature_idx_pairs_) {
    cout << pair.first << " " << pair.second << endl;
  }
*/
}


// 算曲率
void Odom::PrepareSubregion_flat(const PointIRTCloud &scan, const size_t idx_start, const size_t idx_end) {

//  cout << ">>>>>>> " << idx_ring << ", " << idx_start << ", " << idx_end << " <<<<<<<" << endl;
//  const PointIRTCloud &scan = laser_scans_[idx_ring];
  size_t region_size = idx_end - idx_start + 1;
  size_t scan_size = scan.size();
  curvature_idx_pairs_.clear();
  curvature_idx_pairs_.resize(region_size);

  for (size_t i = idx_start, in_region_idx = 0; i <= idx_end; ++i, ++in_region_idx) {

    float point_dist = CalcPointDistance(scan[i]);
    int num_curvature_regions = int(25.0 / point_dist + 0.5) + 1;

    if (i < num_curvature_regions || i + num_curvature_regions >= scan_size) {
      num_curvature_regions = config_.num_curvature_regions_flat;
    }

    int num_point_neighbors = 2 * num_curvature_regions;
    float diff_x = -num_point_neighbors * scan[i].x;
    float diff_y = -num_point_neighbors * scan[i].y;
    float diff_z = -num_point_neighbors * scan[i].z;

    for (int j = 1; j <= num_curvature_regions; ++j) {
      diff_x += scan[i + j].x + scan[i - j].x;
      diff_y += scan[i + j].y + scan[i - j].y;
      diff_z += scan[i + j].z + scan[i - j].z;
    }

    float curvature = (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)/num_point_neighbors;
    pair<float, size_t> curvature_idx_(curvature, i);
    curvature_idx_pairs_[in_region_idx] = curvature_idx_;
  }
  sort(curvature_idx_pairs_.begin(), curvature_idx_pairs_.end());
}


void Odom::MaskPickedInRing(const PointIRTCloud &scan, const size_t in_scan_idx) {
  scan_ring_mask_[in_scan_idx] = 1;

  for (int i = 1; i <= config_.num_feature_regions && in_scan_idx + i < scan.size(); ++i) {
    /// 20cm
    if (CalcSquaredDiff(scan[in_scan_idx + i], scan[in_scan_idx + i - 1]) > 0.04) {
      break;
    }

    scan_ring_mask_[in_scan_idx + i] = 1;
  }

  for (int i = 1; i <= config_.num_feature_regions && in_scan_idx >= i; ++i) {
    if (CalcSquaredDiff(scan[in_scan_idx - i], scan[in_scan_idx - i + 1]) > 0.04) {
      break;
    }

    scan_ring_mask_[in_scan_idx - i] = 1;
  }
}


void Odom::ExtractFeaturePoints() {
  int unstable_pointnum=0;
  int labelCount = 1;
  vector<Eigen::Vector3d> surface_points_normal_temp;
  ///< i is #ring, j is #subregion, k is # in region
  // 一线一线去处理
  for (size_t i = 0; i < row_num_; ++i) {

    size_t start_idx = scan_ranges_[i].first;
    size_t end_idx = scan_ranges_[i].second;

    // skip too short scans
    if (config_.num_curvature_regions_corner < config_.num_curvature_regions_flat) {
      if (end_idx <= start_idx + 2 * config_.num_curvature_regions_flat) {
        continue;
      }
    } else {
      if (end_idx <= start_idx + 2 * config_.num_curvature_regions_corner) {
        continue;
      }
    }

    PointIRTCloud &scan_ring = *laser_scans_[i];
    const vector<art::ImageIndex> &index_ring = image_index_[i];
    size_t scan_size = scan_ring.size();

    // 提取角点
    if (config_.using_sharp_point &&
        config_.lower_ring_num_sharp_point <= (i + 1) && i < config_.upper_ring_num_sharp_point) {

      // 设置掩膜
      PrepareRing_corner(scan_ring);

      // 分区域提取
      for (int j = 0; j < config_.num_scan_subregions; ++j) {
        // 算子区域的索引
        // ((s+d)*N+j*(e-s-d))/N, ((s+d)*N+(j+1)*(e-s-d))/N-1
        size_t sp = ((0 + config_.num_curvature_regions_corner) * (config_.num_scan_subregions - j)
            + (scan_size - config_.num_curvature_regions_corner) * j) / config_.num_scan_subregions;
        size_t ep = ((0 + config_.num_curvature_regions_corner) * (config_.num_scan_subregions - 1 - j)
            + (scan_size - config_.num_curvature_regions_corner) * (j + 1)) / config_.num_scan_subregions - 1;

        // skip empty regions
        if (ep <= sp) {
          continue;
        }

        size_t region_size = ep - sp + 1;

        // extract corner features
        PrepareSubregion_corner(scan_ring, sp, ep);

        int num_largest_picked = 0;
        // 倒序因为要从大曲率的开始
        for (size_t k = region_size; k > 0; --k) {
          // k must be greater than 0
          const pair<float, size_t> &curvature_idx = curvature_idx_pairs_[k - 1];
          float curvature = curvature_idx.first;
          size_t idx = curvature_idx.second;
          size_t in_scan_idx = idx - 0; // scan start index is 0 for all ring scans
          size_t in_region_idx = idx - sp;

          // 标记为地面点的不进行下一步
          if (range_image_[i][image_index_[i][in_scan_idx].col].point_state == -1) {
            continue;
          }

          // 掩膜没有被设置才会进入下一步  并且要符合曲率阈值
          if (scan_ring_mask_[in_scan_idx] == 0 && curvature > config_.sharp_curv_th) {

            vector<ImageIndex> queue_ind;
            ImageIndex index = image_index_[i][in_scan_idx];

            // 对满足曲率的点进行角度阈值聚类  角度越大，两个点离得越近
            if (range_image_[i][index.col].point_state == 0) {

              float d1, d2, alpha, angle;
              int fromIndX, fromIndY, thisIndX, thisIndY;
              bool lineCountFlag[row_num_] = {false};

              // vector<ImageIndex> queue_ind;
              queue_ind.push_back(image_index_[i][in_scan_idx]);
              int queueSize = 1;
              int queueStartInd = 0;

              while (queueSize > 0) {
                // Pop point
                fromIndX = queue_ind[queueStartInd].row;
                fromIndY = queue_ind[queueStartInd].col;
                --queueSize;
                ++queueStartInd;
                // Mark popped point
                range_image_[fromIndX][fromIndY].point_state = labelCount;

                // 深度图上该点周围八个点
                vector<ImageIndex> neighbor;
                neighbor.push_back(ImageIndex{-1, 0});
                neighbor.push_back(ImageIndex{1, 0});
                neighbor.push_back(ImageIndex{0, -1});
                neighbor.push_back(ImageIndex{0, 1});

                neighbor.push_back(ImageIndex{-1, -1});
                neighbor.push_back(ImageIndex{1, -1});
                neighbor.push_back(ImageIndex{0, -1});
                neighbor.push_back(ImageIndex{0, 1});

                // Loop through all the neighboring grids of popped grid
                for (auto iter = neighbor.begin(); iter != neighbor.end(); ++iter) {
                  // new index
                  thisIndX = fromIndX + (*iter).row;
                  thisIndY = fromIndY + (*iter).col;
                  // index should be within the boundary
                  if (thisIndX < 0 || thisIndX >= row_num_) {
                    continue;
                  }
                  // at range image margin (left or right side)
                  // if (thisIndY < 0 || thisIndY >= col_num_)
                  //   continue;
                  if (thisIndY < 0) {
                    thisIndY = col_num_ - 1;
                  }
                  if (thisIndY >= col_num_) {
                    thisIndY = 0;
                  }
                  // prevent infinite loop (caused by put already examined point back)
                  if (range_image_[thisIndX][thisIndY].point_state != 0) {
                    continue;
                  }

                  PointIRT p1 = laser_scans_[fromIndX]->points[range_image_[fromIndX][fromIndY].index];
                  PointIRT p2 = laser_scans_[thisIndX]->points[range_image_[thisIndX][thisIndY].index];

                  Eigen::Vector3d vector_p1{p1.x, p1.y, p1.z};
                  Eigen::Vector3d vector_p2{p2.x, p2.y, p2.z};

                  d1 = std::max(vector_p1.norm(), vector_p2.norm());
                  d2 = std::min(vector_p1.norm(), vector_p2.norm());

                  alpha = acos((vector_p1.dot(vector_p2)) / (vector_p1.norm() * vector_p2.norm()));
                  angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

                  if (angle > 1) {
                    queue_ind.push_back(ImageIndex{thisIndX, thisIndY});
                    ++queueSize;

                    range_image_[thisIndX][thisIndY].point_state = labelCount;
                    lineCountFlag[thisIndX] = true;
                  }
                }
              }

              // check if this segment is valid
              bool feasibleSegment = false;
              int lineCount = 0;
              for (int r = 0; r < row_num_; ++r) {
                if (lineCountFlag[r] == true) {
                  ++lineCount;
                }
              }
              if (lineCount >= 3) {
                feasibleSegment = true;
              }

              if (feasibleSegment == true) {
                ++labelCount;
              } else {
                for (int n = 0; n < queueStartInd; ++n) {
                  // -2表示聚类不稳定的类
                  range_image_[queue_ind[n].row][queue_ind[n].col].point_state = -2;
                  unstable_pointnum++;
                }
              }
            }

            if (range_image_[i][image_index_[i][in_scan_idx].col].point_state < 1) {
              continue;
            }


            if (num_largest_picked < config_.max_corner_less_sharp) {
              ImageIndex index = image_index_[i][in_scan_idx];
              for (int a = 0 ; a < queue_ind.size(); ++a) {
                laser_scans_[queue_ind[a].row]->points[range_image_[queue_ind[a].row][queue_ind[a].col].index].intensity = range_image_[queue_ind[a].row][queue_ind[a].col].point_state;
              }

              if (range_image_[i][index.col].point_state > less_sharp_.size()) {
                less_sharp_.push_back(queue_ind);
              }

              corner_points_less_sharp_.push_back(scan_ring[in_scan_idx]);
              if (num_largest_picked < config_.max_corner_sharp) {
                corner_points_sharp_.push_back(scan_ring[in_scan_idx]);
                range_image_[i][index.col].feature_state = 1;
              }
              MaskPickedInRing(scan_ring, in_scan_idx);
              ++num_largest_picked;
            }

            if (num_largest_picked >= config_.max_corner_less_sharp) {
              break;
            }
          }
        }
      } /// j
    }

    // 如果线束在设置的这个范围内  就提取平面点
    if (config_.using_flat_point &&
        ( (config_.lower_ring_num_x_rot <= (i + 1) && (i + 1) <= config_.upper_ring_num_x_rot) ||
          (config_.lower_ring_num_y_rot <= (i + 1) && (i + 1) <= config_.upper_ring_num_y_rot) ||
          (config_.lower_ring_num_z_trans <= (i + 1) && (i + 1) <= config_.upper_ring_num_z_trans) ||
          (config_.lower_ring_num_z_rot_xy_trans <= (i + 1) && (i + 1) <= config_.lower_ring_num_z_rot_xy_trans) )) {

      // 掩膜
      PrepareRing_flat(scan_ring);

      // extract features from equally sized scan regions
      for (int j = 0; j < config_.num_scan_subregions; ++j) {
        // ((s+d)*N+j*(e-s-d))/N, ((s+d)*N+(j+1)*(e-s-d))/N-1
        size_t sp = ((0 + config_.num_curvature_regions_flat) * (config_.num_scan_subregions - j)
            + (scan_size - config_.num_curvature_regions_flat) * j) / config_.num_scan_subregions;
        size_t ep = ((0 + config_.num_curvature_regions_flat) * (config_.num_scan_subregions - 1 - j)
            + (scan_size - config_.num_curvature_regions_flat) * (j + 1)) / config_.num_scan_subregions - 1;

        // skip empty regions
        if (ep <= sp) {
          continue;
        }
        size_t region_size = ep - sp + 1;

        // 曲率计算 邻域按照平面点设置的来计算
        // extract flat surface features
        PrepareSubregion_flat(scan_ring, sp, ep);

        int num_largest_picked = 0;
        // 正序因为要从小曲率的开始
        for (size_t k = 1; k <= region_size; ++k) {
          const pair<float, size_t> &curvature_idx = curvature_idx_pairs_[k - 1];
          float curvature = curvature_idx.first;

          size_t idx = curvature_idx.second;
          size_t in_scan_idx = idx - 0; // scan start index is 0 for all ring scans
          size_t in_region_idx = idx - sp;

          // 满足平面阈值要求
          if (scan_ring_mask_[in_scan_idx] == 0 && curvature < config_.surf_curv_th) {
            float point_dist = CalcPointDistance(scan_ring[in_scan_idx]);
            // 算了一个邻域大小 不知道是什么原理
            int num_curvature_regions = (point_dist * 0.01 / (DegToRad(config_.deg_diff) * point_dist) + 0.5);

            ImageIndex index = image_index_[i][in_scan_idx];
            PointIRTCloud search_points;

            // 把这一线邻域内的点都加进去
            search_points.push_back(scan_ring[in_scan_idx]);
            for (int c = 1; c <= num_curvature_regions; ++c) {
              if (index.col + c < col_num_) {
                if (range_image_[i][index.col + c].point_state != -3) {
                  search_points.push_back(scan_ring[range_image_[i][index.col + c].index]);                  
                }
              }
              if (index.col >= c) {
                if (range_image_[i][index.col - c].point_state != -3) {
                  search_points.push_back(scan_ring[range_image_[i][index.col - c].index]);
                }
              }
            }

            // 把上一线邻域内的点都加进去
            if (i > 0) {
              if (range_image_[i - 1][index.col].point_state != -3) {
                search_points.push_back(laser_scans_[i - 1]->points[range_image_[i - 1][index.col].index]);
              }
              for (int c = 1; c <= num_curvature_regions; ++c) {
                if (index.col + c < col_num_) {
                  if (range_image_[i - 1][index.col + c].point_state != -3) {
                    search_points.push_back(laser_scans_[i - 1]->points[range_image_[i - 1][index.col + c].index]);                  
                  }
                }
                if (index.col >= c) {
                  if (range_image_[i - 1][index.col - c].point_state != -3) {
                    search_points.push_back(laser_scans_[i - 1]->points[range_image_[i - 1][index.col - c].index]);
                  }
                }
              }
            }

            // 把下一线邻域内的点都加进去
            if (i + 1 < row_num_) {
              if (range_image_[i + 1][index.col].point_state != -3) {
                search_points.push_back(laser_scans_[i + 1]->points[range_image_[i + 1][index.col].index]);
              }
              for (int c = 1; c <= num_curvature_regions; ++c) {
                if (index.col + c < col_num_) {
                  if (range_image_[i + 1][index.col + c].point_state != -3) {
                    search_points.push_back(laser_scans_[i + 1]->points[range_image_[i + 1][index.col + c].index]);                  
                  }
                }
                if (index.col >= c) {
                  if (range_image_[i + 1][index.col - c].point_state != -3) {
                    search_points.push_back(laser_scans_[i + 1]->points[range_image_[i + 1][index.col - c].index]);
                  }
                }
              }
            }

            size_t search_points_num = search_points.size();
            if (search_points_num < 5) {
              continue;
            }

            // 从上面的三线邻域点中选了1m之内的点
            vector<Eigen::Vector3d> near_point;
            for (size_t s = 0; s < search_points_num; ++s) {
              double dis = CalcSquaredDiff(scan_ring[in_scan_idx], search_points[s]);
              if (dis < config_.max_sq_dis) {
                Eigen::Vector3d tmp(search_points[s].x,
                                    search_points[s].y,
                                    search_points[s].z);
                near_point.push_back(tmp);
              }
            }

//            // 协方差矩阵
//            Eigen::Vector3d center(0, 0, 0);
//            for (int s = 0; s < near_point.size(); ++s) {
//              center += near_point[s];
//            }
//            center = center / near_point.size();
//            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
//            for (int s = 0; s < near_point.size(); ++s) {
//              Eigen::Matrix<double, 3, 1> tmpZeroMean = near_point[s] - center;
//              covMat += (tmpZeroMean * tmpZeroMean.transpose());
//            }

//            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

//            // note Eigen library sort eigenvalues in increasing order
//            Eigen::Vector3d unit_direction = saes.eigenvectors().col(0);

            // if (saes.eigenvalues()[1] < 10 * saes.eigenvalues()[0]) {
            //   continue;
            // }

            // 算法线
            Eigen::MatrixXd matA0;
            Eigen::MatrixXd matB0;
            matA0.resize(near_point.size(), 3);
            matB0.resize(near_point.size(), 1);
            for (int s = 0; s < near_point.size(); ++s)
            {
              matA0(s, 0) = near_point[s].x();
              matA0(s, 1) = near_point[s].y();
              matA0(s, 2) = near_point[s].z();
              matB0(s, 0) = -1.0;
            }

            Eigen::Vector3d point_normal = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / point_normal.norm();
            point_normal.normalize();

            bool planeValid = true;
            for (int s = 0; s < near_point.size(); ++s)
            {
              if (fabs( point_normal(0) * near_point[s].x() +
                        point_normal(1) * near_point[s].y() +
                        point_normal(2) * near_point[s].z() + negative_OA_dot_norm) > 0.1)
              {
                planeValid = false;
                break;
              }
            }

            if (planeValid == false) {
              continue;
            }

            PointIRT normal;
            normal.x = point_normal.x();
            normal.y = point_normal.y();
            normal.z = point_normal.z();

            if (num_largest_picked < config_.max_surf_less_flat) {
              surface_points_less_flat_.push_back(scan_ring[in_scan_idx]);
              surface_points_less_flat_index_.push_back(index_ring[in_scan_idx]);
              surface_points_less_flat_normal_.push_back(normal);
              surface_points_normal_temp.push_back(point_normal);
              if (num_largest_picked < config_.max_surf_flat) {
                surface_points_flat_.push_back(scan_ring[in_scan_idx]);
                surface_points_flat_normal_.push_back(normal);
              }
              MaskPickedInRing(scan_ring, in_scan_idx);
              ++num_largest_picked;
            }

            if (num_largest_picked >= config_.max_surf_less_flat) {
              break;
            }
          }
        }
      } /// j
    }
  } /// i
  // 处理完毕

  if (config_.using_corner_point_vector) {
    corner_points_sharp_.clear();
    size_t less_sharp_num = less_sharp_.size();

    for(size_t a = 0; a < less_sharp_num; ++a) {
      size_t point_num = less_sharp_[a].size();

      PointIRTCloud near_point;
      for (size_t b = 0; b < point_num; ++b) {
        ImageIndex index = less_sharp_[a][b];
        near_point.push_back(laser_scans_[index.row]->points[range_image_[index.row][index.col].index]);
      }

      for (size_t b = 0; b < point_num; ++b) {
        ImageIndex index = less_sharp_[a][b];
        if (range_image_[index.row][index.col].feature_state == 1) {
          vector<pair<double, PointIRT>> dis_point;
          for (size_t c = 0; c < point_num; ++c) {
            double dis = CalcSquaredDiff(near_point[b], near_point[c]);
            pair<double, PointIRT> temp(dis, near_point[c]);
            dis_point.push_back(temp);
          }

          sort(dis_point.begin(), dis_point.end(),
            [](const pair<double, PointIRT> &pair_1, const pair<double, PointIRT> &pair_2){
              return pair_1.first < pair_2.first;
            });

          vector<PointIRT> select_near_point;
          vector<bool> row_cout_flag(row_num_, false);
          for (size_t d = 0; d < point_num; ++d) {
            // if (select_near_point.size() > 2 && dis_point[d].first > 0.2) {
            if (select_near_point.size() > 2) {
              break;
            }
            if (row_cout_flag[dis_point[d].second.ring] == false) {
              select_near_point.push_back(dis_point[d].second);
              row_cout_flag[dis_point[d].second.ring] = true;
              // p.points.push_back(dis_point[d].second);
            }
          }

          if (select_near_point.size() > 2) {
            sort(select_near_point.begin(), select_near_point.begin() + 3,
              [](const PointIRT &p_1, const PointIRT &p_2){
                return p_1.ring < p_2.ring;
              });

            Eigen::Vector3d center(0, 0, 0);
            for (int d = 0; d < select_near_point.size(); ++d) {
              Eigen::Vector3d temp(select_near_point[d].x, select_near_point[d].y, select_near_point[d].z);
              center += temp;
            }
            center = center / select_near_point.size();

            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            for (int d = 0; d < select_near_point.size(); ++d) {
              Eigen::Vector3d temp(select_near_point[d].x, select_near_point[d].y, select_near_point[d].z); 

              Eigen::Matrix<double, 3, 1> tmpZeroMean = temp - center;
              covMat += (tmpZeroMean * tmpZeroMean.transpose());
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

            // note Eigen library sort eigenvalues in increasing order
            Eigen::Vector3d v = saes.eigenvalues();
            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2).normalized();

            Eigen::Vector3d p_0(select_near_point[0].x, select_near_point[0].y, select_near_point[0].z);
            Eigen::Vector3d p_1(select_near_point[1].x, select_near_point[1].y, select_near_point[1].z);
            Eigen::Vector3d p_2(select_near_point[2].x, select_near_point[2].y, select_near_point[2].z);

            double angle = acos((p_0 - p_1).dot(p_2 - p_1) / (p_0 - p_1).norm() / (p_2 - p_1).norm()) / M_PI * 180;
            // LOG(INFO) << angle << " " << select_near_point.size();

            if (angle > 150) {
              PointIRT temp;
              temp.x = unit_direction.x();
              temp.y = unit_direction.y();
              temp.z = unit_direction.z();
              temp.timestamp = dis_point[0].second.timestamp;
              corner_points_vector_.push_back(temp);
              corner_points_sharp_.push_back(dis_point[0].second);
            }
          }
        }
      }
    }
  }


  if (config_.using_surf_point_normal) {
    surface_points_flat_.clear();
    surface_points_flat_normal_.clear();

    pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> pair_nomal;
    size_t surf_points_less_flat_num = surface_points_less_flat_.size();

    vector<pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>>> vector_nomal_z_trans;
    vector<pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>>> vector_nomal_x_rot;
    vector<pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>>> vector_nomal_y_rot;
    vector<pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>>> vector_nomal_z_rot_xy_trans;

    for (size_t n = 0; n < surf_points_less_flat_num; ++n) {
      pair_nomal.first =  surface_points_normal_temp[n];
      pair_nomal.second.first = surface_points_less_flat_[n];
      pair_nomal.second.second = surface_points_less_flat_index_[n];

      int scan_id = surface_points_less_flat_[n].ring;
      if (config_.lower_ring_num_z_trans <= (scan_id + 1) && (scan_id + 1) <= config_.upper_ring_num_z_trans) {
        vector_nomal_z_trans.push_back(pair_nomal);
      }
      if (config_.lower_ring_num_x_rot <= (scan_id + 1) && (scan_id + 1) <= config_.upper_ring_num_x_rot) {
        vector_nomal_x_rot.push_back(pair_nomal);
      }
      if (config_.lower_ring_num_y_rot <= (scan_id + 1) && (scan_id + 1) <= config_.upper_ring_num_y_rot) {
        vector_nomal_y_rot.push_back(pair_nomal);
      }
      if (config_.lower_ring_num_z_rot_xy_trans <= (scan_id + 1) && (scan_id + 1)<= config_.upper_ring_num_z_rot_xy_trans) {
        vector_nomal_z_rot_xy_trans.push_back(pair_nomal);
      }
    }

    size_t num_z_trans = vector_nomal_z_trans.size();
    size_t num_x_rot = vector_nomal_x_rot.size();
    size_t num_y_rot = vector_nomal_y_rot.size();
    size_t num_z_rot_xy_trans = vector_nomal_z_rot_xy_trans.size();

    sort(vector_nomal_z_rot_xy_trans.begin(), vector_nomal_z_rot_xy_trans.end(), 
      [](const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_1, const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_2){
        return fabs(pair_1.first.dot(Eigen::Vector3d::UnitX())) > fabs(pair_2.first.dot(Eigen::Vector3d::UnitX()));
      });

    double con_trs_x = 0;
    for (int i = 0, j = 0; i < num_z_rot_xy_trans && j < config_.flat_extract_num_x_trans; ++i, ++j) {
      PointIRT point_normal_temp;
      point_normal_temp.x = vector_nomal_z_rot_xy_trans[i].first.x();
      point_normal_temp.y = vector_nomal_z_rot_xy_trans[i].first.y();
      point_normal_temp.z = vector_nomal_z_rot_xy_trans[i].first.z();
      if (range_image_[vector_nomal_z_rot_xy_trans[i].second.second.row][vector_nomal_z_rot_xy_trans[i].second.second.col].feature_state == 0) {
        surface_points_flat_normal_.push_back(point_normal_temp);
        surface_points_flat_.push_back(vector_nomal_z_rot_xy_trans[i].second.first);
        surface_points_flat_z_rot_xy_trans_.push_back(vector_nomal_z_rot_xy_trans[i].second.first);
        range_image_[vector_nomal_z_rot_xy_trans[i].second.second.row][vector_nomal_z_rot_xy_trans[i].second.second.col].feature_state = 1;
      }
    }


    sort(vector_nomal_z_rot_xy_trans.begin(), vector_nomal_z_rot_xy_trans.end(), 
      [](const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_1, const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_2){
        return fabs(pair_1.first.dot(Eigen::Vector3d::UnitY())) > fabs(pair_2.first.dot(Eigen::Vector3d::UnitY()));
      });

    double con_trs_y = 0;
    for (int i = 0, j = 0; i < num_z_rot_xy_trans && j < config_.flat_extract_num_y_trans; ++i, ++j) {
      PointIRT point_normal_temp;
      point_normal_temp.x = vector_nomal_z_rot_xy_trans[i].first.x();
      point_normal_temp.y = vector_nomal_z_rot_xy_trans[i].first.y();
      point_normal_temp.z = vector_nomal_z_rot_xy_trans[i].first.z();
      if (range_image_[vector_nomal_z_rot_xy_trans[i].second.second.row][vector_nomal_z_rot_xy_trans[i].second.second.col].feature_state == 0) {
        surface_points_flat_normal_.push_back(point_normal_temp);
        surface_points_flat_.push_back(vector_nomal_z_rot_xy_trans[i].second.first);
        surface_points_flat_z_rot_xy_trans_.push_back(vector_nomal_z_rot_xy_trans[i].second.first);
        range_image_[vector_nomal_z_rot_xy_trans[i].second.second.row][vector_nomal_z_rot_xy_trans[i].second.second.col].feature_state = 1;
      }
    }


    sort(vector_nomal_z_rot_xy_trans.begin(), vector_nomal_z_rot_xy_trans.end(),
      [](const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>>&pair_1, const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_2){
        Eigen::Vector3d point_temp_1, point_temp_2;
        point_temp_1.x() = pair_1.second.first.x;
        point_temp_1.y() = pair_1.second.first.y;
        point_temp_1.z() = pair_1.second.first.z;

        point_temp_2.x() = pair_2.second.first.x;
        point_temp_2.y() = pair_2.second.first.y;
        point_temp_2.z() = pair_2.second.first.z;

        double corss_z_1 = pair_1.first.cross(point_temp_1).dot(Eigen::Vector3d::UnitZ());
        double corss_z_2 = pair_2.first.cross(point_temp_2).dot(Eigen::Vector3d::UnitZ());
        return fabs(corss_z_1) > fabs(corss_z_2);
      });

    double con_rot_z = 0;
    for (int i = 0, j = 0; i < num_z_rot_xy_trans && j < config_.flat_extract_num_z_rot; ++i, ++j) {
      PointIRT point_normal_temp;
      point_normal_temp.x = vector_nomal_z_rot_xy_trans[i].first.x();
      point_normal_temp.y = vector_nomal_z_rot_xy_trans[i].first.y();
      point_normal_temp.z = vector_nomal_z_rot_xy_trans[i].first.z();
      if (range_image_[vector_nomal_z_rot_xy_trans[i].second.second.row][vector_nomal_z_rot_xy_trans[i].second.second.col].feature_state == 0) {
        surface_points_flat_normal_.push_back(point_normal_temp);
        surface_points_flat_.push_back(vector_nomal_z_rot_xy_trans[i].second.first);
        surface_points_flat_z_rot_xy_trans_.push_back(vector_nomal_z_rot_xy_trans[i].second.first);
        range_image_[vector_nomal_z_rot_xy_trans[i].second.second.row][vector_nomal_z_rot_xy_trans[i].second.second.col].feature_state = 1;
      }
    }


    sort(vector_nomal_z_trans.begin(), vector_nomal_z_trans.end(), 
      [](const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_1, const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_2){
        return fabs(pair_1.first.dot(Eigen::Vector3d::UnitZ())) > fabs(pair_2.first.dot(Eigen::Vector3d::UnitZ()));
      });

    double con_trs_z = 0;
    for (int i = 0, j = 0; i < num_z_trans && j < config_.flat_extract_num_z_trans; ++i, ++j) {
      PointIRT point_normal_temp;
      point_normal_temp.x = vector_nomal_z_trans[i].first.x();
      point_normal_temp.y = vector_nomal_z_trans[i].first.y();
      point_normal_temp.z = vector_nomal_z_trans[i].first.z();
      if (range_image_[vector_nomal_z_trans[i].second.second.row][vector_nomal_z_trans[i].second.second.col].feature_state == 0) {
        surface_points_flat_normal_.push_back(point_normal_temp);
        surface_points_flat_.push_back(vector_nomal_z_trans[i].second.first);
        surface_points_flat_z_trans_.push_back(vector_nomal_z_trans[i].second.first);
        con_trs_z += fabs(vector_nomal_z_trans[i].first.z());
        range_image_[vector_nomal_z_trans[i].second.second.row][vector_nomal_z_trans[i].second.second.col].feature_state = 1;
      }
    }


    sort(vector_nomal_x_rot.begin(), vector_nomal_x_rot.end(),
      [](const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_1, const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_2){
        Eigen::Vector3d point_temp_1, point_temp_2;
        point_temp_1.x() = pair_1.second.first.x;
        point_temp_1.y() = pair_1.second.first.y;
        point_temp_1.z() = pair_1.second.first.z;

        point_temp_2.x() = pair_2.second.first.x;
        point_temp_2.y() = pair_2.second.first.y;
        point_temp_2.z() = pair_2.second.first.z;

        double corss_x_1 = pair_1.first.cross(point_temp_1).dot(Eigen::Vector3d::UnitX());
        double corss_x_2 = pair_2.first.cross(point_temp_2).dot(Eigen::Vector3d::UnitX());
        return fabs(corss_x_1) > fabs(corss_x_2);
      });

    double con_rot_x = 0;
    for (int i = 0, j = 0; i < num_x_rot && j < config_.flat_extract_num_x_rot; ++i, ++j) {
      PointIRT point_normal_temp;
      point_normal_temp.x = vector_nomal_x_rot[i].first.x();
      point_normal_temp.y = vector_nomal_x_rot[i].first.y();
      point_normal_temp.z = vector_nomal_x_rot[i].first.z();
      if (range_image_[vector_nomal_x_rot[i].second.second.row][vector_nomal_x_rot[i].second.second.col].feature_state == 0) {
        surface_points_flat_normal_.push_back(point_normal_temp);
        surface_points_flat_.push_back(vector_nomal_x_rot[i].second.first);
        surface_points_flat_x_rot_.push_back(vector_nomal_x_rot[i].second.first);
        range_image_[vector_nomal_x_rot[i].second.second.row][vector_nomal_x_rot[i].second.second.col].feature_state = 1;
      }
    }


    sort(vector_nomal_y_rot.begin(), vector_nomal_y_rot.end(),
      [](const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_1, const pair<Eigen::Vector3d, pair<PointIRT, ImageIndex>> &pair_2){
        Eigen::Vector3d point_temp_1, point_temp_2;
        point_temp_1.x() = pair_1.second.first.x;
        point_temp_1.y() = pair_1.second.first.y;
        point_temp_1.z() = pair_1.second.first.z;

        point_temp_2.x() = pair_2.second.first.x;
        point_temp_2.y() = pair_2.second.first.y;
        point_temp_2.z() = pair_2.second.first.z;

        double corss_y_1 = pair_1.first.cross(point_temp_1).dot(Eigen::Vector3d::UnitY());
        double corss_y_2 = pair_2.first.cross(point_temp_2).dot(Eigen::Vector3d::UnitY());
        return fabs(corss_y_1) > fabs(corss_y_2);
      });

    double con_rot_y = 0;
    for (int i = 0, j = 0; i < num_y_rot && j < config_.flat_extract_num_y_rot; ++i, ++j) {
      PointIRT point_normal_temp;
      point_normal_temp.x = vector_nomal_y_rot[i].first.x();
      point_normal_temp.y = vector_nomal_y_rot[i].first.y();
      point_normal_temp.z = vector_nomal_y_rot[i].first.z();
      if (range_image_[vector_nomal_y_rot[i].second.second.row][vector_nomal_y_rot[i].second.second.col].feature_state == 0) {
        surface_points_flat_normal_.push_back(point_normal_temp);
        surface_points_flat_.push_back(vector_nomal_y_rot[i].second.first);
        surface_points_flat_y_rot_.push_back(vector_nomal_y_rot[i].second.first);
        range_image_[vector_nomal_y_rot[i].second.second.row][vector_nomal_y_rot[i].second.second.col].feature_state = 1;
      }
    }
  }
  LOG(INFO) << "corner_points_sharp_num: " << corner_points_sharp_.size();
  LOG(INFO) << "surface_points_flat_num: " << surface_points_flat_.size();
  LOG(INFO) << "corner_points_less_sharp_num: " << corner_points_less_sharp_.size();
  LOG(INFO) << "surface_points_less_flat_num: " << surface_points_less_flat_.size();
} // ExtractFeaturePoints


void Odom::TransformToEnd(const PointIRT &pi, PointIRT &po) {
  float s = 1.0 - pi.timestamp / config_.scan_period;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity().slerp(s, Eigen::Quaterniond(transf_last_curr_.inverse().rotation()));
  Eigen::Vector3d t = s * transf_last_curr_.inverse().translation();
  Eigen::Vector3d point(pi.x, pi.y, pi.z);
  Eigen::Vector3d un_point = q * point + t;

  po.x = un_point.x();
  po.y = un_point.y();
  po.z = un_point.z();
  po.intensity = pi.intensity;
  po.timestamp = pi.timestamp;
  po.ring = pi.ring;
}


void Odom::RotateVectToEnd(const PointIRT &pi, PointIRT &po) {
  float s = 1.0 - pi.timestamp / config_.scan_period;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity().slerp(s, Eigen::Quaterniond(transf_last_curr_.inverse().rotation()));
  Eigen::Vector3d point(pi.x, pi.y, pi.z);
  Eigen::Vector3d un_point = q * point;

  po.x = un_point.x();
  po.y = un_point.y();
  po.z = un_point.z();
}


void Odom::FrontEndForSLAM() {
  static vector<pair<pair<int, Eigen::Affine3d>, pair<Eigen::Affine3d, PointIRTCloud>>> transf_cloud_vec;
  PointIRTCloud temp_key_frame_cloud;

  size_t corner_points_sharp_num = corner_points_sharp_.size();
  size_t surface_points_flat_num = surface_points_flat_.size();
  size_t corner_points_less_sharp_num = corner_points_less_sharp_.size();
  size_t surface_points_less_flat_num = surface_points_less_flat_.size();

  static Eigen::Affine3d estimate_pose_last;
  bool key_frame = false;
  if (!system_inited_) {
//    transf_cloud_vec.clear();
    system_inited_ = true;
    key_frame = true;
    local_map_.Reset(30);
    trans_sum_for_local_map_.Reset(30);
    estimate_pose_last.setIdentity();
    estimate_pose_ = estimate_pose_last;
  } else {
    Eigen::Map<Eigen::Vector3d> so3(para_so3_);
    Eigen::Map<Eigen::Vector3d> t(para_t_);

    //匀速模型
    estimate_pose_ = estimate_pose_last * transf_last_curr_;
    so3 = Sophus::SO3<double>(estimate_pose_.rotation()).log();
    t = estimate_pose_.translation();

    PointIRTCloudPtr corner_points_sharp_ptr(new PointIRTCloud(corner_points_sharp_));
    PointIRTCloudPtr surface_points_flat_ptr(new PointIRTCloud(surface_points_flat_));
    PointIRTCloudPtr corner_points_vector_ptr(new PointIRTCloud(corner_points_vector_));
    PointIRTCloudPtr surface_points_flat_normal_ptr(new PointIRTCloud(surface_points_flat_normal_));

//    // kitti不需要去除畸变
//    for (size_t n = 0; n < corner_points_sharp_num; ++n) {
//      TransformToEnd(corner_points_sharp_ptr->points[n], corner_points_sharp_ptr->points[n]);
////      RotateVectToEnd(corner_points_vector_ptr->points[n], corner_points_vector_ptr->points[n]);
//    }

//    for (size_t n = 0; n < surface_points_flat_num; ++n) {
//      TransformToEnd(surface_points_flat_ptr->points[n], surface_points_flat_ptr->points[n]);
//      RotateVectToEnd(surface_points_flat_normal_ptr->points[n], surface_points_flat_normal_ptr->points[n]);
//    }

    for (size_t opti_counter = 0; opti_counter < 5; ++opti_counter) {
      int corner_correspondence = 0;
      int plane_correspondence = 0;

      ceres::LossFunction *loss_function_corner =  new ceres::CauchyLoss(0.1);
      ceres::LossFunction *loss_function_surface =  new ceres::CauchyLoss(0.1);
      ceres::LocalParameterization *so3_parameterization = new SE3Parameterization();
      ceres::Problem::Options problem_options;

      ceres::Problem problem(problem_options);
      problem.AddParameterBlock(para_so3_, 3, so3_parameterization);
      problem.AddParameterBlock(para_t_, 3);

      std::vector<int> point_search_ind;
      std::vector<float> point_search_sq_dis;

      if (config_.using_sharp_point) {
        PointIRTCloud transformed_cloud;
        pcl::transformPointCloud(*corner_points_sharp_ptr, transformed_cloud, estimate_pose_);

        for (size_t i = 0; i < corner_points_sharp_num; ++i) {
          kdtree_local_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis);
          if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
            Eigen::Vector3d curr_point(corner_points_sharp_ptr->points[i].x,
                                       corner_points_sharp_ptr->points[i].y,
                                       corner_points_sharp_ptr->points[i].z);

            Eigen::Vector3d near_point(local_map_cloud_ptr_->points[point_search_ind[0]].x,
                                       local_map_cloud_ptr_->points[point_search_ind[0]].y,
                                       local_map_cloud_ptr_->points[point_search_ind[0]].z);

//            Eigen::Vector3d curr_point_vect(corner_points_vector_ptr->points[i].x,
//                                            corner_points_vector_ptr->points[i].y,
//                                            corner_points_vector_ptr->points[i].z);

            ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, near_point, so3, t);
            // ceres::CostFunction *cost_function = LidarEdgeVectorFactor::Create(curr_point, near_point, curr_point_vect, so3, t);
//            ceres::CostFunction *cost_function = LidarEdgeVectorFactorZRotXYTrans::Create(curr_point, near_point, curr_point_vect, so3, t);
            problem.AddResidualBlock(cost_function, loss_function_corner, para_so3_, para_t_);
            corner_correspondence++;
          }
        }
      }

      if (config_.using_flat_point) {
        PointIRTCloud transformed_cloud;
        pcl::transformPointCloud(*surface_points_flat_ptr, transformed_cloud, estimate_pose_);

        for (size_t i = 0; i < surface_points_flat_num; ++i) {
          kdtree_local_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis);
          if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
            Eigen::Vector3d curr_point(surface_points_flat_ptr->points[i].x,
                                       surface_points_flat_ptr->points[i].y,
                                       surface_points_flat_ptr->points[i].z);

            Eigen::Vector3d near_point(local_map_cloud_ptr_->points[point_search_ind[0]].x,
                                       local_map_cloud_ptr_->points[point_search_ind[0]].y,
                                       local_map_cloud_ptr_->points[point_search_ind[0]].z);

            Eigen::Vector3d curr_point_norm(surface_points_flat_normal_ptr->points[i].x,
                                            surface_points_flat_normal_ptr->points[i].y,
                                            surface_points_flat_normal_ptr->points[i].z);
            ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, near_point, curr_point_norm, so3, t);
            problem.AddResidualBlock(cost_function, loss_function_corner, para_so3_, para_t_);
            plane_correspondence++;
          }
        }
      }

      if ((corner_correspondence + plane_correspondence) < 10) {
          std::cout << "less correspondence! *************************************************\n";
      }

      ceres::Solver::Options options;
      options.num_threads = 6;
      options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
      options.gradient_tolerance = 1e-15;
      options.function_tolerance = 1e-15;
      options.linear_solver_type = ceres::DENSE_QR;
      // options.max_num_iterations = 10;
      options.minimizer_progress_to_stdout = false;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      estimate_pose_.linear() = Sophus::SO3<double>::exp(so3).matrix();
      estimate_pose_.translation() = t;
    }

    size_t key_frame_num = trans_sum_for_local_map_.size();
    Eigen::Affine3d last_key_frame_trans = trans_sum_for_local_map_[key_frame_num - 1];

    Eigen::Affine3d transf_delt = last_key_frame_trans.inverse() * estimate_pose_;
    double t_delt = transf_delt.translation().norm();
    double q_delt = Eigen::Quaterniond(transf_delt.rotation()).angularDistance(Eigen::Quaterniond::Identity()) * 180.0 / M_PI;

    if (t_delt > 2.0 || q_delt > 5.0) {
      key_frame = true;
    }
  }
  transf_last_curr_ = estimate_pose_last.inverse() * estimate_pose_;
  estimate_pose_last = estimate_pose_;

  if (key_frame) {
    PointIRTCloud corner_points_less_sharp = corner_points_less_sharp_;
    PointIRTCloud surface_points_less_flat = surface_points_less_flat_;

    // kitti不需要去除畸变
//    for (size_t n = 0; n < corner_points_less_sharp_num; ++n) {
//      TransformToEnd(corner_points_less_sharp.points[n], corner_points_less_sharp.points[n]);
//    }
//    for (size_t n = 0; n < surface_points_less_flat_num; ++n) {
//      TransformToEnd(surface_points_less_flat.points[n], surface_points_less_flat.points[n]);
//    }

    corner_points_less_sharp += surface_points_less_flat;
    local_map_.push(corner_points_less_sharp);
    trans_sum_for_local_map_.push(estimate_pose_);

    local_map_cloud_ptr_->clear();
    for (size_t i = 0; i < local_map_.size(); ++i) {
      PointIRTCloud transformed_cloud;
      pcl::transformPointCloud(local_map_[i], transformed_cloud, trans_sum_for_local_map_[i]);
      *local_map_cloud_ptr_ += transformed_cloud;
    }

    pcl::VoxelGrid<PointIRT> down_size_filter;
    down_size_filter.setInputCloud(local_map_cloud_ptr_);
    down_size_filter.setLeafSize(0.4, 0.4, 0.4);
    down_size_filter.filter(*local_map_cloud_ptr_);
    kdtree_local_map_->setInputCloud(local_map_cloud_ptr_);
  } else {
//    PointIRTCloud corner_points_less_sharp = corner_points_less_sharp_;
//    PointIRTCloud surface_points_less_flat = surface_points_less_flat_;

////    for (size_t n = 0; n < corner_points_less_sharp_num; ++n) {
////      TransformToEnd(corner_points_less_sharp.points[n], corner_points_less_sharp.points[n]);
////    }
////    for (size_t n = 0; n < surface_points_less_flat_num; ++n) {
////      TransformToEnd(surface_points_less_flat.points[n], surface_points_less_flat.points[n]);
////    }
//    corner_points_less_sharp += surface_points_less_flat;
//    pcl::transformPointCloud(corner_points_less_sharp, corner_points_less_sharp, estimate_pose_);

//    local_map_cloud_ptr_->clear();
//    for (size_t i = 0; i < local_map_.size(); ++i) {
//      PointIRTCloud transformed_cloud;
//      pcl::transformPointCloud(local_map_[i], transformed_cloud, trans_sum_for_local_map_[i]);
//      *local_map_cloud_ptr_ += transformed_cloud;
//    }
//    *local_map_cloud_ptr_ += corner_points_less_sharp;

//    pcl::VoxelGrid<PointIRT> down_size_filter;
//    down_size_filter.setInputCloud(local_map_cloud_ptr_);
//    down_size_filter.setLeafSize(0.1, 0.1, 0.1);
//    down_size_filter.filter(*local_map_cloud_ptr_);
//    kdtree_local_map_->setInputCloud(local_map_cloud_ptr_);
  }
  std::cout << "local map point num:" << local_map_cloud_ptr_->size() << std::endl;

}


void Odom::FrontEndForMapping() {

  Eigen::Map<Eigen::Vector3d> so3(para_so3_);
  Eigen::Map<Eigen::Vector3d> t(para_t_);

  so3 = Sophus::SO3<double>(estimate_pose_.rotation()).log();
  t = estimate_pose_.translation();

  size_t corner_points_sharp_num = corner_points_sharp_.size();
  size_t surface_points_flat_num = surface_points_flat_.size();
  size_t corner_points_less_sharp_num = corner_points_less_sharp_.size();
  size_t surface_points_less_flat_num = surface_points_less_flat_.size();

  PointIRTCloudPtr corner_points_sharp_ptr(new PointIRTCloud(corner_points_sharp_));
  PointIRTCloudPtr surface_points_flat_ptr(new PointIRTCloud(surface_points_flat_));
  PointIRTCloudPtr corner_points_vector_ptr(new PointIRTCloud(corner_points_vector_));
  PointIRTCloudPtr surface_points_flat_normal_ptr(new PointIRTCloud(surface_points_flat_normal_));

  for (size_t n = 0; n < corner_points_sharp_num; ++n) {
    TransformToEnd(corner_points_sharp_ptr->points[n], corner_points_sharp_ptr->points[n]);
    // RotateVectToEnd(corner_points_vector_ptr->points[n], corner_points_vector_ptr->points[n]);
  }
  for (size_t n = 0; n < surface_points_flat_num; ++n) {
    TransformToEnd(surface_points_flat_ptr->points[n], surface_points_flat_ptr->points[n]);
    RotateVectToEnd(surface_points_flat_normal_ptr->points[n], surface_points_flat_normal_ptr->points[n]);
  }

  static Eigen::Affine3d estimate_pose_last;
  bool key_frame = false;
  if (!system_inited_) {
    system_inited_ = true;
    key_frame = true;
    estimate_pose_last = estimate_pose_;
    local_map_.Reset(30);
    trans_sum_for_local_map_.Reset(30);
  } else {
    for (size_t opti_counter = 0; opti_counter < 5; ++opti_counter) {

      int corner_correspondence = 0;
      int plane_correspondence = 0;

      ceres::LossFunction *loss_function_corner =  new ceres::CauchyLoss(0.1);
      ceres::LossFunction *loss_function_surface =  new ceres::CauchyLoss(0.1);
      ceres::LocalParameterization *so3_parameterization = new SE3Parameterization();

      ceres::Problem::Options problem_options;
      ceres::Problem problem(problem_options);

      problem.AddParameterBlock(para_so3_, 3, so3_parameterization);
      problem.AddParameterBlock(para_t_, 3);
      // problem.SetParameterBlockConstant(para_so3_);
      problem.SetParameterBlockConstant(para_t_);

      std::vector<int> point_search_ind;
      std::vector<float> point_search_sq_dis;

      if (config_.using_sharp_point) {
        PointIRTCloud transformed_cloud;
        pcl::transformPointCloud(*corner_points_sharp_ptr, transformed_cloud, estimate_pose_);

        for (size_t i = 0; i < corner_points_sharp_num; ++i) {
          kdtree_local_map_->nearestKSearch(transformed_cloud.points[i], 1, point_search_ind, point_search_sq_dis);
          if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
            Eigen::Vector3d curr_point(corner_points_sharp_ptr->points[i].x,
                                        corner_points_sharp_ptr->points[i].y,
                                        corner_points_sharp_ptr->points[i].z);

            Eigen::Vector3d near_point(local_map_cloud_ptr_->points[point_search_ind[0]].x,
                                       local_map_cloud_ptr_->points[point_search_ind[0]].y,
                                       local_map_cloud_ptr_->points[point_search_ind[0]].z);
                    
            // Eigen::Vector3d curr_point_vect(corner_points_vector_ptr->points[i].x,
            //                                 corner_points_vector_ptr->points[i].y,
            //                                 corner_points_vector_ptr->points[i].z);

            ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, near_point, so3, t);
            // ceres::CostFunction *cost_function = LidarEdgeVectorFactor::Create(curr_point, near_point, curr_point_vect, so3, t);
            // ceres::CostFunction *cost_function = LidarEdgeVectorFactorZRotXYTrans::Create(curr_point, near_point, curr_point_vect, so3, t);
            problem.AddResidualBlock(cost_function, loss_function_corner, para_so3_, para_t_);

            corner_correspondence++;
          }
        }
      }

      if (config_.using_flat_point) {
        PointIRTCloud transformed_cloud;
        pcl::transformPointCloud(*surface_points_flat_ptr, transformed_cloud, estimate_pose_);

        for (size_t i = 0; i < surface_points_flat_num; ++i) {
          kdtree_local_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis); 
          if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
            Eigen::Vector3d curr_point(surface_points_flat_ptr->points[i].x,
                                        surface_points_flat_ptr->points[i].y,
                                        surface_points_flat_ptr->points[i].z);

            Eigen::Vector3d near_point(local_map_cloud_ptr_->points[point_search_ind[0]].x,
                                        local_map_cloud_ptr_->points[point_search_ind[0]].y,
                                        local_map_cloud_ptr_->points[point_search_ind[0]].z);

            Eigen::Vector3d curr_point_norm(surface_points_flat_normal_ptr->points[i].x,
                                            surface_points_flat_normal_ptr->points[i].y,
                                            surface_points_flat_normal_ptr->points[i].z);

            ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, near_point, curr_point_norm, so3, t);
            problem.AddResidualBlock(cost_function, loss_function_corner, para_so3_, para_t_);
            plane_correspondence++;
          }
        }
      }

      if ((corner_correspondence + plane_correspondence) < 10) {
        LOG(INFO) << "less correspondence! *************************************************";
      }

      ceres::Solver::Options options;
      options.num_threads = 6;
      options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
      options.gradient_tolerance = 1e-15;
      options.function_tolerance = 1e-15;
      options.linear_solver_type = ceres::DENSE_QR;
      options.minimizer_progress_to_stdout = false;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      estimate_pose_.linear() = Sophus::SO3<double>::exp(so3).matrix();;
      estimate_pose_.translation() = t;
    }

    size_t key_frame_num = trans_sum_for_local_map_.size();
    Eigen::Affine3d last_key_frame_trans = trans_sum_for_local_map_[key_frame_num - 1];    

    Eigen::Affine3d transf_delt = last_key_frame_trans.inverse() * estimate_pose_;
    double t_delt = transf_delt.translation().norm();
    double q_delt = Eigen::Quaterniond(transf_delt.rotation()).angularDistance(Eigen::Quaterniond::Identity()) * 180.0 / M_PI;

    if (t_delt > 2.0 || q_delt > 5.0) {
      key_frame = true;
    }
  }

  transf_last_curr_ = estimate_pose_last.inverse() * estimate_pose_;
  estimate_pose_last = estimate_pose_;

  if (key_frame) {
    PointIRTCloud corner_points_less_sharp = corner_points_less_sharp_;
    PointIRTCloud surface_points_less_flat = surface_points_less_flat_;

    for (size_t n = 0; n < corner_points_less_sharp_num; ++n) {
      TransformToEnd(corner_points_less_sharp.points[n], corner_points_less_sharp.points[n]);
    }
    for (size_t n = 0; n < surface_points_less_flat_num; ++n) {
      TransformToEnd(surface_points_less_flat.points[n], surface_points_less_flat.points[n]);
    }

    corner_points_less_sharp += surface_points_less_flat;
    local_map_.push(corner_points_less_sharp);
    trans_sum_for_local_map_.push(estimate_pose_);

    local_map_cloud_ptr_->clear();
    for (size_t i = 0; i < local_map_.size(); ++i) {
      PointIRTCloud transformed_cloud;
      pcl::transformPointCloud(local_map_[i], transformed_cloud, trans_sum_for_local_map_[i]);
      *local_map_cloud_ptr_ += transformed_cloud;
    }
    kdtree_local_map_->setInputCloud(local_map_cloud_ptr_);
  }
}


void Odom::FrontEndForMapping(const std::deque<MapFrame> &local_map_frames) {
  static Eigen::Affine3d estimate_pose_last;
  bool key_frame = false;
  if (!system_inited_) {
    system_inited_ = true;
    local_map_.Reset(30);
    trans_sum_for_local_map_.Reset(30);
    local_map_cloud_ptr_->clear();

    size_t local_map_num = local_map_frames.size();
    if (local_map_num <= local_map_.capacity()) {
      for (size_t i = 0; i < local_map_num; ++i) {
        local_map_.push(local_map_frames[i].feature_points);
        trans_sum_for_local_map_.push(local_map_frames[i].transform);
      }
    } else {
      for (size_t i = local_map_num - local_map_.capacity(); i < local_map_num; ++i) {
        local_map_.push(local_map_frames[i].feature_points);
        trans_sum_for_local_map_.push(local_map_frames[i].transform);
      }
    }

    for (size_t i = 0; i < local_map_.size(); ++i) {
      PointIRTCloud transformed_cloud;
      pcl::transformPointCloud(local_map_[i], transformed_cloud, trans_sum_for_local_map_[i]);
      *local_map_cloud_ptr_ += transformed_cloud;
    }

    pcl::VoxelGrid<PointIRT> down_size_filter;
    down_size_filter.setInputCloud(local_map_cloud_ptr_);
    down_size_filter.setLeafSize(0.1, 0.1, 0.1);
    down_size_filter.filter(*local_map_cloud_ptr_);
    kdtree_local_map_->setInputCloud(local_map_cloud_ptr_);
    estimate_pose_last = estimate_pose_;
  }

  Eigen::Map<Eigen::Vector3d> so3(para_so3_);
  Eigen::Map<Eigen::Vector3d> t(para_t_);

  estimate_pose_ = estimate_pose_last * transf_last_curr_;
  so3 = Sophus::SO3<double>(estimate_pose_.rotation()).log();
  t = estimate_pose_.translation();

  PointIRTCloudPtr corner_points_sharp_ptr(new PointIRTCloud(corner_points_sharp_));
  PointIRTCloudPtr surface_points_flat_ptr(new PointIRTCloud(surface_points_flat_));
  PointIRTCloudPtr corner_points_vector_ptr(new PointIRTCloud(corner_points_vector_));
  PointIRTCloudPtr surface_points_flat_normal_ptr(new PointIRTCloud(surface_points_flat_normal_));

  size_t corner_points_sharp_num = corner_points_sharp_.size();
  size_t surface_points_flat_num = surface_points_flat_.size();
  size_t corner_points_less_sharp_num = corner_points_less_sharp_.size();
  size_t surface_points_less_flat_num = surface_points_less_flat_.size();

  for (size_t n = 0; n < corner_points_sharp_num; ++n) {
    TransformToEnd(corner_points_sharp_ptr->points[n], corner_points_sharp_ptr->points[n]);
    RotateVectToEnd(corner_points_vector_ptr->points[n], corner_points_vector_ptr->points[n]);
  }

  for (size_t n = 0; n < surface_points_flat_num; ++n) {
    TransformToEnd(surface_points_flat_ptr->points[n], surface_points_flat_ptr->points[n]);
    RotateVectToEnd(surface_points_flat_normal_ptr->points[n], surface_points_flat_normal_ptr->points[n]);
  }

  for (size_t opti_counter = 0; opti_counter < 5; ++opti_counter) {
    int corner_correspondence = 0;
    int plane_correspondence = 0;

    ceres::LossFunction *loss_function_corner =  new ceres::CauchyLoss(0.1);
    ceres::LossFunction *loss_function_surface =  new ceres::CauchyLoss(0.1);
    ceres::LocalParameterization *so3_parameterization = new SE3Parameterization();
    ceres::Problem::Options problem_options;

    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(para_so3_, 3, so3_parameterization);
    problem.AddParameterBlock(para_t_, 3);

    std::vector<int> point_search_ind;
    std::vector<float> point_search_sq_dis;

    if (config_.using_sharp_point) {
      PointIRTCloud transformed_cloud;
      pcl::transformPointCloud(*corner_points_sharp_ptr, transformed_cloud, estimate_pose_);

      for (size_t i = 0; i < corner_points_sharp_num; ++i) {
        kdtree_local_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis);
        if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
          Eigen::Vector3d curr_point(corner_points_sharp_ptr->points[i].x,
                                     corner_points_sharp_ptr->points[i].y,
                                     corner_points_sharp_ptr->points[i].z);

          Eigen::Vector3d near_point(local_map_cloud_ptr_->points[point_search_ind[0]].x,
                                     local_map_cloud_ptr_->points[point_search_ind[0]].y,
                                     local_map_cloud_ptr_->points[point_search_ind[0]].z);

          // Eigen::Vector3d curr_point_vect(corner_points_vector_ptr->points[i].x,
          //                                 corner_points_vector_ptr->points[i].y,
          //                                 corner_points_vector_ptr->points[i].z);

          ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, near_point, so3, t);
          // ceres::CostFunction *cost_function = LidarEdgeVectorFactor::Create(curr_point, near_point, curr_point_vect, so3, t);
          problem.AddResidualBlock(cost_function, loss_function_corner, para_so3_, para_t_);
          corner_correspondence++;
        }
      }
    }

    if (config_.using_flat_point) {
      PointIRTCloud transformed_cloud;
      pcl::transformPointCloud(*surface_points_flat_ptr, transformed_cloud, estimate_pose_);

      for (size_t i = 0; i < surface_points_flat_num; ++i) {
        kdtree_local_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis); 
        if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
          Eigen::Vector3d curr_point(surface_points_flat_ptr->points[i].x,
                                     surface_points_flat_ptr->points[i].y,
                                     surface_points_flat_ptr->points[i].z);

          Eigen::Vector3d near_point(local_map_cloud_ptr_->points[point_search_ind[0]].x,
                                     local_map_cloud_ptr_->points[point_search_ind[0]].y,
                                     local_map_cloud_ptr_->points[point_search_ind[0]].z);

          Eigen::Vector3d curr_point_norm(surface_points_flat_normal_ptr->points[i].x,
                                          surface_points_flat_normal_ptr->points[i].y,
                                          surface_points_flat_normal_ptr->points[i].z);
          ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, near_point, curr_point_norm, so3, t);
          problem.AddResidualBlock(cost_function, loss_function_corner, para_so3_, para_t_);
          plane_correspondence++;
        }
      }
    }

    if ((corner_correspondence + plane_correspondence) < 10) {
        std::cout << "less correspondence! *************************************************\n";
    }

    ceres::Solver::Options options;
    options.num_threads = 6;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.gradient_tolerance = 1e-15;
    options.function_tolerance = 1e-15;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    estimate_pose_.linear() = Sophus::SO3<double>::exp(so3).matrix();;
    estimate_pose_.translation() = t;
  }

  transf_last_curr_ = estimate_pose_last.inverse() * estimate_pose_;
  estimate_pose_last = estimate_pose_;

  size_t key_frame_num = trans_sum_for_local_map_.size();
  Eigen::Affine3d last_key_frame_trans = trans_sum_for_local_map_[key_frame_num - 1];

  Eigen::Affine3d transf_delt = last_key_frame_trans.inverse() * estimate_pose_;
  double t_delt = transf_delt.translation().norm();
  double q_delt = Eigen::Quaterniond(transf_delt.rotation()).angularDistance(Eigen::Quaterniond::Identity()) * 180.0 / M_PI;

  if (t_delt > 2.0 || q_delt > 5.0) {
    key_frame = true;
  }

  if (key_frame) {
    PointIRTCloud corner_points_less_sharp = corner_points_less_sharp_;
    PointIRTCloud surface_points_less_flat = surface_points_less_flat_;

    for (size_t n = 0; n < corner_points_less_sharp_num; ++n) {
      TransformToEnd(corner_points_less_sharp.points[n], corner_points_less_sharp.points[n]);
    }
    for (size_t n = 0; n < surface_points_less_flat_num; ++n) {
      TransformToEnd(surface_points_less_flat.points[n], surface_points_less_flat.points[n]);
    }

    corner_points_less_sharp += surface_points_less_flat;
    local_map_.push(corner_points_less_sharp);
    trans_sum_for_local_map_.push(estimate_pose_);

    local_map_cloud_ptr_->clear();
    for (size_t i = 0; i < local_map_.size(); ++i) {
      PointIRTCloud transformed_cloud;
      pcl::transformPointCloud(local_map_[i], transformed_cloud, trans_sum_for_local_map_[i]);
      *local_map_cloud_ptr_ += transformed_cloud;
    }

    pcl::VoxelGrid<PointIRT> down_size_filter;
    down_size_filter.setInputCloud(local_map_cloud_ptr_);
    down_size_filter.setLeafSize(0.1, 0.1, 0.1);
    down_size_filter.filter(*local_map_cloud_ptr_);
    kdtree_local_map_->setInputCloud(local_map_cloud_ptr_);
  }
}


void Odom::setInputTarget(const PointIRTCloud &map_cloud) {
  map_cloud_ptr_->clear();
  *map_cloud_ptr_ += map_cloud;
  kdtree_map_->setInputCloud(map_cloud_ptr_);
}


void Odom::SystemInital() {
  system_inited_ = false;
};


void Odom::FrontEndForLocalization() {

  Eigen::Map<Eigen::Quaterniond> q(para_q_);
  Eigen::Map<Eigen::Vector3d> so3(para_so3_);
  Eigen::Map<Eigen::Vector3d> t(para_t_);

  q = Eigen::Quaterniond(estimate_pose_.rotation());
  so3 = Sophus::SO3<double>(estimate_pose_.rotation()).log();
  t = estimate_pose_.translation();

  size_t corner_points_sharp_num = corner_points_sharp_.size();
  size_t surface_points_flat_num = surface_points_flat_.size();
  size_t corner_points_less_sharp_num = corner_points_less_sharp_.size();
  size_t surface_points_less_flat_num = surface_points_less_flat_.size();

  // LOG(INFO) << corner_points_sharp_num;
  // LOG(INFO) << surface_points_flat_num;
  // LOG(INFO) << corner_points_less_sharp_num;
  // LOG(INFO) << surface_points_less_flat_num;

  // sensor_msgs::PointCloud2 sharp_cloud;
  // pcl::toROSMsg(corner_points_sharp_, sharp_cloud);
  // sharp_cloud.header.stamp = ros::Time::now();
  // sharp_cloud.header.frame_id = "/map";
  // pub_sharp_cloud_.publish(sharp_cloud);

  sensor_msgs::PointCloud2 flat_cloud;
  pcl::toROSMsg(surface_points_flat_, flat_cloud);
  flat_cloud.header.stamp = ros::Time::now();
  flat_cloud.header.frame_id = "/map";
  pub_flat_cloud_.publish(flat_cloud);

  // sensor_msgs::PointCloud2 less_sharp_cloud;
  // pcl::toROSMsg(corner_points_less_sharp_, less_sharp_cloud);
  // less_sharp_cloud.header.stamp = ros::Time::now();
  // less_sharp_cloud.header.frame_id = "/map";
  // pub_less_sharp_cloud_.publish(less_sharp_cloud);

  sensor_msgs::PointCloud2 less_flat_cloud;
  pcl::toROSMsg(surface_points_less_flat_, less_flat_cloud);
  less_flat_cloud.header.stamp = ros::Time::now();
  less_flat_cloud.header.frame_id = "/map";
  pub_less_flat_cloud_.publish(less_flat_cloud);

  PointIRTCloudPtr corner_points_sharp_ptr(new PointIRTCloud(corner_points_sharp_));
  PointIRTCloudPtr surface_points_flat_ptr(new PointIRTCloud(surface_points_flat_));
  PointIRTCloudPtr corner_points_vector_ptr(new PointIRTCloud(corner_points_vector_));
  PointIRTCloudPtr surface_points_flat_normal_ptr(new PointIRTCloud(surface_points_flat_normal_));

  for (size_t n = 0; n < corner_points_sharp_num; ++n) {
    TransformToEnd(corner_points_sharp_ptr->points[n], corner_points_sharp_ptr->points[n]);
    RotateVectToEnd(corner_points_vector_ptr->points[n], corner_points_vector_ptr->points[n]);
  }

  for (size_t n = 0; n < surface_points_flat_num; ++n) {
    TransformToEnd(surface_points_flat_ptr->points[n], surface_points_flat_ptr->points[n]);
    RotateVectToEnd(surface_points_flat_normal_ptr->points[n], surface_points_flat_normal_ptr->points[n]);
  }

  bool key_frame = false;
  if (!system_inited_) {
    key_frame = true;
    local_map_.Reset(30);
    trans_sum_for_local_map_.Reset(30);
  } else {
    for (size_t opti_counter = 0; opti_counter < 5; ++opti_counter) {

      int corner_correspondence = 0;
      int plane_correspondence = 0;

      ceres::LossFunction *loss_function_corner =  new ceres::CauchyLoss(0.1);
      ceres::LossFunction *loss_function_surface =  new ceres::CauchyLoss(0.1);
      ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
      ceres::LocalParameterization *so3_parameterization = new SE3Parameterization();
      
      ceres::Problem::Options problem_options;
      ceres::Problem problem(problem_options);

      problem.AddParameterBlock(para_q_, 4, q_parameterization);
      problem.AddParameterBlock(para_so3_, 3, so3_parameterization);
      problem.AddParameterBlock(para_t_, 3);

      std::vector<int> point_search_ind;
      std::vector<float> point_search_sq_dis;

      if (config_.using_sharp_point) {
        PointIRTCloud transformed_cloud;
        pcl::transformPointCloud(*corner_points_sharp_ptr, transformed_cloud, estimate_pose_);

        for (size_t i = 0; i < corner_points_sharp_num; ++i) {
          kdtree_local_map_->nearestKSearch(transformed_cloud.points[i], 1, point_search_ind, point_search_sq_dis);
          if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
            Eigen::Vector3d curr_point(corner_points_sharp_ptr->points[i].x,
                                       corner_points_sharp_ptr->points[i].y,
                                       corner_points_sharp_ptr->points[i].z);

            Eigen::Vector3d near_point(local_map_cloud_ptr_->points[point_search_ind[0]].x,
                                       local_map_cloud_ptr_->points[point_search_ind[0]].y,
                                       local_map_cloud_ptr_->points[point_search_ind[0]].z);

            Eigen::Vector3d curr_point_vect(corner_points_vector_ptr->points[i].x,
                                            corner_points_vector_ptr->points[i].y,
                                            corner_points_vector_ptr->points[i].z);

            ceres::CostFunction *cost_function = LidarDistanceFactorOld::Create(curr_point, near_point, 1.0);
            // ceres::CostFunction *cost_function = LidarEdgeVectorFactorOld::Create(curr_point, near_point, curr_point_vect, 1.0);
            problem.AddResidualBlock(cost_function, loss_function_corner, para_q_, para_t_);

            // ceres::CostFunction *cost_function = LidarDistanceFactorTest::Create(curr_point, near_point);
            // // ceres::CostFunction *cost_function = LidarEdgeVectorFactor::Create(curr_point, near_point, curr_point_vect, so3, t);
            // problem.AddResidualBlock(cost_function, loss_function_corner, para_so3_, para_t_);

            corner_correspondence++;
          }
        }
      }

      if (config_.using_flat_point) {
        PointIRTCloud transformed_cloud;
        pcl::transformPointCloud(*surface_points_flat_ptr, transformed_cloud, estimate_pose_);

        for (size_t i = 0; i < surface_points_flat_num; ++i) {
          kdtree_local_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis); 
          if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
            Eigen::Vector3d curr_point(surface_points_flat_ptr->points[i].x,
                                        surface_points_flat_ptr->points[i].y,
                                        surface_points_flat_ptr->points[i].z);

            Eigen::Vector3d near_point(local_map_cloud_ptr_->points[point_search_ind[0]].x,
                                        local_map_cloud_ptr_->points[point_search_ind[0]].y,
                                        local_map_cloud_ptr_->points[point_search_ind[0]].z);

            Eigen::Vector3d curr_point_norm(surface_points_flat_normal_ptr->points[i].x,
                                            surface_points_flat_normal_ptr->points[i].y,
                                            surface_points_flat_normal_ptr->points[i].z);

            ceres::CostFunction *cost_function = LidarPlaneNormFactorOld::Create(curr_point, near_point, curr_point_norm, 1.0);
            problem.AddResidualBlock(cost_function, loss_function_surface, para_q_, para_t_);

            // ceres::CostFunction *cost_function = LidarPlaneNormFactorTest::Create(curr_point, near_point, curr_point_norm);
            // problem.AddResidualBlock(cost_function, loss_function_surface, para_so3_, para_t_);

            plane_correspondence++;
          }
        }
      }


      // // *************tight coupling*************
      // if (config_.using_sharp_point) {
      //   PointIRTCloud transformed_cloud;
      //   pcl::transformPointCloud(*corner_points_sharp_ptr, transformed_cloud, estimate_pose_);

      //   for (size_t i = 0; i < corner_points_sharp_num; ++i) {
      //     kdtree_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis);
      //     if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
      //       Eigen::Vector3d curr_point(corner_points_sharp_ptr->points[i].x,
      //                                   corner_points_sharp_ptr->points[i].y,
      //                                   corner_points_sharp_ptr->points[i].z);

      //       Eigen::Vector3d near_point(map_cloud_ptr_->points[point_search_ind[0]].x,
      //                                   map_cloud_ptr_->points[point_search_ind[0]].y,
      //                                   map_cloud_ptr_->points[point_search_ind[0]].z);

      //       Eigen::Vector3d curr_point_vect(corner_points_vector_ptr->points[i].x,
      //                                       corner_points_vector_ptr->points[i].y,
      //                                       corner_points_vector_ptr->points[i].z);

      //       // ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, near_point, so3, t);
      //       ceres::CostFunction *cost_function = LidarEdgeVectorFactor::Create(curr_point, near_point, curr_point_vect, so3, t);
      //       problem.AddResidualBlock(cost_function, loss_function_corner, para_so3_, para_t_);
      //       corner_correspondence++;
      //     }
      //   }
      // }

      // if (config_.using_flat_point) {
      //   PointIRTCloud transformed_cloud;
      //   pcl::transformPointCloud(*surface_points_flat_ptr, transformed_cloud, estimate_pose_);

      //   for (size_t i = 0; i < surface_points_flat_num; ++i) {
      //     kdtree_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis);
      //     if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
      //       Eigen::Vector3d curr_point(surface_points_flat_ptr->points[i].x,
      //                                   surface_points_flat_ptr->points[i].y,
      //                                   surface_points_flat_ptr->points[i].z);

      //       Eigen::Vector3d near_point(map_cloud_ptr_->points[point_search_ind[0]].x,
      //                                  map_cloud_ptr_->points[point_search_ind[0]].y,
      //                                  map_cloud_ptr_->points[point_search_ind[0]].z);

      //       Eigen::Vector3d curr_point_norm(surface_points_flat_normal_ptr->points[i].x,
      //                                       surface_points_flat_normal_ptr->points[i].y,
      //                                       surface_points_flat_normal_ptr->points[i].z);
      //       ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, near_point, curr_point_norm, so3, t);
      //       problem.AddResidualBlock(cost_function, loss_function_surface, para_so3_, para_t_);
      //       plane_correspondence++;                                    
      //     }
      //   }
      // }

      if ((corner_correspondence + plane_correspondence) < 10) {
        LOG(INFO) << "less correspondence! *************************************************";
      }

      ceres::Solver::Options options;
      options.num_threads = 6;
      options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
      options.gradient_tolerance = 1e-15;
      options.function_tolerance = 1e-15;
      options.linear_solver_type = ceres::DENSE_QR;
      options.minimizer_progress_to_stdout = false;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      Eigen::Affine3d estimate_pose_update;
      estimate_pose_update.linear() = q.toRotationMatrix();
      // estimate_pose_update.linear() = Sophus::SO3<double>::exp(so3).matrix();;
      estimate_pose_update.translation() = t;

      Eigen::Affine3d delt_transf;
      delt_transf = estimate_pose_.inverse() * estimate_pose_update;
      transf_last_curr_ = transf_last_curr_ * delt_transf;

      estimate_pose_ = estimate_pose_update;
    }

    size_t key_frame_num = trans_sum_for_local_map_.size();
    Eigen::Affine3d last_key_frame_trans = trans_sum_for_local_map_[key_frame_num - 1];    

    Eigen::Affine3d transf_delt = last_key_frame_trans.inverse() * estimate_pose_;
    double t_delt = transf_delt.translation().norm();
    double q_delt = Eigen::Quaterniond(transf_delt.rotation()).angularDistance(Eigen::Quaterniond::Identity()) * 180.0 / M_PI;

    if (t_delt > 2.0 || q_delt > 5.0) {
      key_frame = true;
    }
  }


  // // *************tight coupling*************
  // if (!system_inited_) {
  //   system_inited_ = true;
  //   // corner_points_sharp_ptr.reset(new PointIRTCloud(corner_points_sharp_));
  //   // surface_points_flat_ptr.reset(new PointIRTCloud(surface_points_flat_));
  //   // surface_points_flat_normal_ptr.reset(new PointIRTCloud(surface_points_flat_normal_));

  //   // for (size_t n = 0; n < corner_points_sharp_num; ++n) {
  //   //   TransformToEnd(corner_points_sharp_ptr->points[n], corner_points_sharp_ptr->points[n]);
  //   // }

  //   // for (size_t n = 0; n < surface_points_flat_num; ++n) {
  //   //   TransformToEnd(surface_points_flat_ptr->points[n], surface_points_flat_ptr->points[n]);
  //   //   RotateVectToEnd(surface_points_flat_normal_ptr->points[n], surface_points_flat_normal_ptr->points[n]);
  //   // }

  //   for (int opti_counter = 0; opti_counter < 5; ++opti_counter) {
  //     int corner_correspondence = 0;
  //     int plane_correspondence = 0;

  //     ceres::LossFunction *loss_function_corner =  new ceres::CauchyLoss(0.1);
  //     ceres::LossFunction *loss_function_surface =  new ceres::CauchyLoss(0.1);
  //     ceres::LocalParameterization *so3_parameterization = new SE3Parameterization();

  //     ceres::Problem::Options problem_options;
  //     ceres::Problem problem(problem_options);

  //     problem.AddParameterBlock(para_so3_, 3, so3_parameterization);
  //     problem.AddParameterBlock(para_t_, 3);

  //     std::vector<int> point_search_ind;
  //     std::vector<float> point_search_sq_dis;

  //     if (config_.using_sharp_point) {
  //       PointIRTCloud transformed_cloud;
  //       pcl::transformPointCloud(*corner_points_sharp_ptr, transformed_cloud, estimate_pose_);

  //       for (size_t i = 0; i < corner_points_sharp_num; ++i) {
  //         kdtree_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis);
  //         if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
  //           Eigen::Vector3d curr_point(corner_points_sharp_ptr->points[i].x,
  //                                      corner_points_sharp_ptr->points[i].y,
  //                                      corner_points_sharp_ptr->points[i].z);

  //           Eigen::Vector3d near_point(map_cloud_ptr_->points[point_search_ind[0]].x,
  //                                      map_cloud_ptr_->points[point_search_ind[0]].y,
  //                                      map_cloud_ptr_->points[point_search_ind[0]].z);

  //           Eigen::Vector3d curr_point_vect(corner_points_vector_ptr->points[i].x,
  //                                           corner_points_vector_ptr->points[i].y,
  //                                           corner_points_vector_ptr->points[i].z);

  //           // ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, near_point, so3, t);
  //           ceres::CostFunction *cost_function = LidarEdgeVectorFactor::Create(curr_point, near_point, curr_point_vect, so3, t);

  //           problem.AddResidualBlock(cost_function, loss_function_corner, para_so3_, para_t_);
  //           corner_correspondence++;
  //         }
  //       }
  //     }

  //     // find correspondence for plane features
  //     if (config_.using_flat_point) {
  //       PointIRTCloud transformed_cloud;
  //       pcl::transformPointCloud(*surface_points_flat_ptr, transformed_cloud, estimate_pose_);

  //       for (size_t i = 0; i < surface_points_flat_num; ++i) {
  //         kdtree_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis);
  //         if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
  //           Eigen::Vector3d curr_point(surface_points_flat_ptr->points[i].x,
  //                                      surface_points_flat_ptr->points[i].y,
  //                                      surface_points_flat_ptr->points[i].z);

  //           Eigen::Vector3d near_point(map_cloud_ptr_->points[point_search_ind[0]].x,
  //                                      map_cloud_ptr_->points[point_search_ind[0]].y,
  //                                      map_cloud_ptr_->points[point_search_ind[0]].z);

  //           Eigen::Vector3d curr_point_norm(surface_points_flat_normal_ptr->points[i].x,
  //                                           surface_points_flat_normal_ptr->points[i].y,
  //                                           surface_points_flat_normal_ptr->points[i].z);
  //           ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, near_point, curr_point_norm, so3, t);
  //           problem.AddResidualBlock(cost_function, loss_function_surface, para_so3_, para_t_);
  //           plane_correspondence++;                                    
  //         }
  //       }
  //     }

  //     if ((corner_correspondence + plane_correspondence) < 10) {
  //       LOG(INFO) << "less correspondence! *************************************************";
  //     }

  //     ceres::Solver::Options options;
  //     options.num_threads = 6;
  //     options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  //     options.gradient_tolerance = 1e-15;
  //     options.function_tolerance = 1e-15;
  //     options.linear_solver_type = ceres::DENSE_QR;
  //     // options.max_num_iterations = 10;
  //     options.minimizer_progress_to_stdout = false;
  //     ceres::Solver::Summary summary;
  //     ceres::Solve(options, &problem, &summary);

  //     Eigen::Affine3d estimate_pose_update;
  //     estimate_pose_update.linear() = Sophus::SO3<double>::exp(so3).matrix();;
  //     estimate_pose_update.translation() = t;
  //     estimate_pose_ = estimate_pose_update;
  //   }
  // }




  // *************loose coupling*************
  corner_points_sharp_ptr.reset(new PointIRTCloud(corner_points_sharp_));
  surface_points_flat_ptr.reset(new PointIRTCloud(surface_points_flat_));
  corner_points_vector_ptr.reset(new PointIRTCloud(corner_points_vector_));
  surface_points_flat_normal_ptr.reset(new PointIRTCloud(surface_points_flat_normal_));

  for (size_t n = 0; n < corner_points_sharp_num; ++n) {
    TransformToEnd(corner_points_sharp_ptr->points[n], corner_points_sharp_ptr->points[n]);
    RotateVectToEnd(corner_points_vector_ptr->points[n], corner_points_vector_ptr->points[n]);
  }

  for (size_t n = 0; n < surface_points_flat_num; ++n) {
    TransformToEnd(surface_points_flat_ptr->points[n], surface_points_flat_ptr->points[n]);
    RotateVectToEnd(surface_points_flat_normal_ptr->points[n], surface_points_flat_normal_ptr->points[n]);
  }

  for (int opti_counter = 0; opti_counter < 5; ++opti_counter) {
    int corner_correspondence = 0;
    int plane_correspondence = 0;

    ceres::LossFunction *loss_function_corner =  new ceres::CauchyLoss(0.1);
    ceres::LossFunction *loss_function_surface =  new ceres::CauchyLoss(0.1);
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    ceres::LocalParameterization *so3_parameterization = new SE3Parameterization();

    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    problem.AddParameterBlock(para_q_, 4, q_parameterization);
    // problem.AddParameterBlock(para_so3_, 3, so3_parameterization);
    problem.AddParameterBlock(para_t_, 3);

    std::vector<int> point_search_ind;
    std::vector<float> point_search_sq_dis;

    if (config_.using_sharp_point) {
      PointIRTCloud transformed_cloud;
      pcl::transformPointCloud(*corner_points_sharp_ptr, transformed_cloud, estimate_pose_);

      for (size_t i = 0; i < corner_points_sharp_num; ++i) {
        kdtree_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis);
        if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
          Eigen::Vector3d curr_point(corner_points_sharp_ptr->points[i].x,
                                      corner_points_sharp_ptr->points[i].y,
                                      corner_points_sharp_ptr->points[i].z);

          Eigen::Vector3d near_point(map_cloud_ptr_->points[point_search_ind[0]].x,
                                     map_cloud_ptr_->points[point_search_ind[0]].y,
                                     map_cloud_ptr_->points[point_search_ind[0]].z);

          Eigen::Vector3d curr_point_vect(corner_points_vector_ptr->points[i].x,
                                          corner_points_vector_ptr->points[i].y,
                                          corner_points_vector_ptr->points[i].z);

          // ceres::CostFunction *cost_function = LidarDistanceFactorOld::Create(curr_point, near_point, 1.0);
            ceres::CostFunction *cost_function = LidarEdgeVectorFactorOld::Create(curr_point, near_point, curr_point_vect, 1.0);
          problem.AddResidualBlock(cost_function, loss_function_corner, para_q_, para_t_);

          // ceres::CostFunction *cost_function = LidarDistanceFactorTest::Create(curr_point, near_point);
          // // ceres::CostFunction *cost_function = LidarEdgeVectorFactor::Create(curr_point, near_point, curr_point_vect, so3, t);
          // problem.AddResidualBlock(cost_function, loss_function_corner, para_so3_, para_t_);

          corner_correspondence++;
        }
      }
    }

// find correspondence for plane features
    if (config_.using_flat_point) {
      PointIRTCloud transformed_cloud;
      pcl::transformPointCloud(*surface_points_flat_ptr, transformed_cloud, estimate_pose_);

      for (size_t i = 0; i < surface_points_flat_num; ++i) {
        kdtree_map_->nearestKSearch(transformed_cloud[i], 1, point_search_ind, point_search_sq_dis);
        if (point_search_sq_dis[0] < config_.distance_sq_threshold) {
          Eigen::Vector3d curr_point(surface_points_flat_ptr->points[i].x,
                                      surface_points_flat_ptr->points[i].y,
                                      surface_points_flat_ptr->points[i].z);

          Eigen::Vector3d near_point(map_cloud_ptr_->points[point_search_ind[0]].x,
                                     map_cloud_ptr_->points[point_search_ind[0]].y,
                                     map_cloud_ptr_->points[point_search_ind[0]].z);

          Eigen::Vector3d curr_point_norm(surface_points_flat_normal_ptr->points[i].x,
                                          surface_points_flat_normal_ptr->points[i].y,
                                          surface_points_flat_normal_ptr->points[i].z);

          ceres::CostFunction *cost_function = LidarPlaneNormFactorOld::Create(curr_point, near_point, curr_point_norm, 1.0);
          problem.AddResidualBlock(cost_function, loss_function_surface, para_q_, para_t_);

          // ceres::CostFunction *cost_function = LidarPlaneNormFactorTest::Create(curr_point, near_point, curr_point_norm);
          // problem.AddResidualBlock(cost_function, loss_function_surface, para_so3_, para_t_);

          plane_correspondence++;                                    
        }
      }
    }

    if ((corner_correspondence + plane_correspondence) < 10) {
      LOG(INFO) << "less correspondence! *************************************************";
    }

    ceres::Solver::Options options;
    options.num_threads = 6;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.gradient_tolerance = 1e-15;
    options.function_tolerance = 1e-15;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Affine3d estimate_pose_update;
    estimate_pose_update.linear() = q.toRotationMatrix();
    // estimate_pose_update.linear() = Sophus::SO3<double>::exp(so3).matrix();;
    estimate_pose_update.translation() = t;

    if (!system_inited_) {
      system_inited_ = true;
    } else {
      Eigen::Affine3d delt_transf;
      delt_transf = estimate_pose_.inverse() * estimate_pose_update;
      transf_last_curr_ = transf_last_curr_ * delt_transf;
    }

    estimate_pose_ = estimate_pose_update;
  }
  static int num = 0;
  num += 1;
  static PointIRTCloud map;
  if (key_frame) {
    PointIRTCloud corner_points_less_sharp = corner_points_less_sharp_;
    PointIRTCloud surface_points_less_flat = surface_points_less_flat_;

    for (size_t n = 0; n < corner_points_less_sharp_num; ++n) {
      TransformToEnd(corner_points_less_sharp.points[n], corner_points_less_sharp.points[n]);
    }
    for (size_t n = 0; n < surface_points_less_flat_num; ++n) {
      TransformToEnd(surface_points_less_flat.points[n], surface_points_less_flat.points[n]);
    }

    corner_points_less_sharp += surface_points_less_flat;
    local_map_.push(corner_points_less_sharp);
    trans_sum_for_local_map_.push(estimate_pose_);
    PointIRTCloud transf_points;
    pcl::transformPointCloud(corner_points_less_sharp, transf_points, estimate_pose_);
    map += transf_points;
    local_map_cloud_ptr_->clear();
    for (size_t i = 0; i < local_map_.size(); ++i) {
      PointIRTCloud transformed_cloud;
      pcl::transformPointCloud(local_map_[i], transformed_cloud, trans_sum_for_local_map_[i]);
      *local_map_cloud_ptr_ += transformed_cloud;
    }
    kdtree_local_map_->setInputCloud(local_map_cloud_ptr_);
  }

  // if (num == 880) {
  //   pcl::io::savePCDFileBinary("/home/cdx/Desktop/map_loca.pcd", map);
  // }
}


void Odom::BackEndForLoop(std::vector<Eigen::Affine3d> &frame_poses, const Eigen::Affine3d &loop_pose) {
  size_t frame_num = frame_poses.size() - 1;
  double paraso3arry[frame_num][3];
  double paratarry[frame_num][3];
  for(size_t i = 0; i < frame_num; ++i) {
    Eigen::Map<Eigen::Vector3d> so3(paraso3arry[i]);
    Eigen::Map<Eigen::Vector3d> t(paratarry[i]);

    so3 = Sophus::SO3<double>(frame_poses[i + 1].rotation()).log();
    t = frame_poses[i + 1].translation();
  }

  for (size_t opti_counter = 0; opti_counter < 1; ++opti_counter) {

    ceres::LossFunction *loss_function =  new ceres::CauchyLoss(0.1);
    ceres::LocalParameterization *so3_parameterization = new SE3Parameterization();

    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    for (size_t i = 0; i < frame_num; ++i) {
      problem.AddParameterBlock(paraso3arry[i], 3, so3_parameterization);
      problem.AddParameterBlock(paratarry[i], 3);
    }

    for (size_t i = 0; i < frame_num - 1; ++i) {
      Eigen::Vector3d pretonextso3 = Sophus::SO3<double>((frame_poses[i+2].inverse() * frame_poses[i+1]).rotation()).log();
      Eigen::Vector3d pretonextt((frame_poses[i+2].inverse() * frame_poses[i+1]).translation());

      ceres::CostFunction *cost_function = EndBackFactor::Create(pretonextso3, pretonextt);
      problem.AddResidualBlock(cost_function, loss_function, paraso3arry[i], paratarry[i], paraso3arry[i+1], paratarry[i+1]);
    }

    Eigen::Vector3d first_frame_so3 = Sophus::SO3<double>(frame_poses[0].rotation()).log();
    Eigen::Vector3d first_frame_t(frame_poses[0].translation());

    Eigen::Vector3d fir_to_sec_so3 = Sophus::SO3<double>((frame_poses[1].inverse() * frame_poses[0]).rotation()).log();
    Eigen::Vector3d fir_to_sec_t = (frame_poses[1].inverse() * frame_poses[0]).translation();

    ceres::CostFunction *cost_function_first_frame = EndBackFirstFrameFactor::Create(fir_to_sec_so3, fir_to_sec_t, first_frame_so3, first_frame_t);
    problem.AddResidualBlock(cost_function_first_frame, loss_function, paraso3arry[0], paratarry[0]);

    //loop_constraint
    Eigen::Vector3d fir_to_loop_so3 = Sophus::SO3<double>((loop_pose.inverse() * frame_poses[0]).rotation()).log();
    Eigen::Vector3d fir_to_loop_t = (loop_pose.inverse() * frame_poses[0]).translation();

    ceres::CostFunction *cost_function_loop_frame = EndBackFirstFrameFactor::Create(fir_to_loop_so3, fir_to_loop_t, first_frame_so3, first_frame_t);
    problem.AddResidualBlock(cost_function_loop_frame, loss_function, paraso3arry[frame_num-1], paratarry[frame_num - 1]);

    ceres::Solver::Options options;
    options.num_threads = 6;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.gradient_tolerance = 1e-10;
    options.function_tolerance = 1e-10;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
  }

  for(size_t i = 0; i < frame_num; ++i) {
    frame_poses[i + 1].linear() = Sophus::SO3<double>::exp(Eigen::Vector3d(paraso3arry[i])).matrix();
    frame_poses[i + 1].translation() = Eigen::Vector3d(paratarry[i]);
  }
}


void Odom::BackEndForGNSS(std::vector<Eigen::Affine3d> &frame_poses, const Eigen::Affine3d &gnss_pose) {
  size_t frame_num = frame_poses.size() - 2;
  double paraso3arry[frame_num][4];
  double paratarry[frame_num][3];
  for(size_t i = 0; i < frame_num; ++i) {
    Eigen::Map<Eigen::Vector3d> so3(paraso3arry[i]);
    Eigen::Map<Eigen::Vector3d> t(paratarry[i]);

    so3 = Sophus::SO3<double>(frame_poses[i + 1].rotation()).log();
    t = frame_poses[i + 1].translation();
  }

  for (size_t opti_counter = 0; opti_counter < 1; ++opti_counter) {

    ceres::LossFunction *loss_function =  new ceres::CauchyLoss(0.1);
    ceres::LocalParameterization *so3_parameterization = new SE3Parameterization();

    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    for (size_t i = 0; i < frame_num; ++i) {
      problem.AddParameterBlock(paraso3arry[i], 3, so3_parameterization);
      problem.AddParameterBlock(paratarry[i], 3);
    }

    for (size_t i = 0; i < frame_num - 1; ++i) {
      Eigen::Vector3d pretonextso3 = Sophus::SO3<double>((frame_poses[i+2].inverse() * frame_poses[i+1]).rotation()).log();
      Eigen::Vector3d pretonextt((frame_poses[i+2].inverse() * frame_poses[i+1]).translation());

      ceres::CostFunction *cost_function = EndBackFactor::Create(pretonextso3, pretonextt);
      problem.AddResidualBlock(cost_function, loss_function, paraso3arry[i], paratarry[i], paraso3arry[i+1], paratarry[i+1]);
    }

    Eigen::Vector3d first_frame_so3 = Sophus::SO3<double>(frame_poses[0].rotation()).log();
    Eigen::Vector3d first_frame_t(frame_poses[0].translation());

    Eigen::Vector3d fir_to_sec_so3 = Sophus::SO3<double>((frame_poses[1].inverse() * frame_poses[0]).rotation()).log();
    Eigen::Vector3d fir_to_sec_t = (frame_poses[1].inverse() * frame_poses[0]).translation();

    ceres::CostFunction *cost_function_first_frame = EndBackFirstFrameFactor::Create(fir_to_sec_so3, fir_to_sec_t, first_frame_so3, first_frame_t);
    problem.AddResidualBlock(cost_function_first_frame, loss_function, paraso3arry[0], paratarry[0]);

    Eigen::Vector3d gnss_pose_so3 = Sophus::SO3<double>(gnss_pose.rotation()).log();
    Eigen::Vector3d gnss_pose_t(gnss_pose.translation());

    Eigen::Vector3d pre_to_last_so3 = Sophus::SO3<double>((frame_poses.back().inverse() * *(frame_poses.end()-2)).rotation()).log();
    Eigen::Vector3d pre_to_last_t((frame_poses.back().inverse() * *(frame_poses.end()-2)).translation());

    ceres::CostFunction *cost_function_lastfream = EndBackLastFrameFactor::Create(pre_to_last_so3, pre_to_last_t, gnss_pose_so3, gnss_pose_t);
    problem.AddResidualBlock(cost_function_lastfream, loss_function, paraso3arry[frame_num-1], paratarry[frame_num-1]);

    ceres::Solver::Options options;
    options.num_threads = 6;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.gradient_tolerance = 1e-10;
    options.function_tolerance = 1e-10;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
  }

  for(size_t i = 0; i < frame_num; ++i) {
    frame_poses[i + 1].linear() = Sophus::SO3<double>::exp(Eigen::Vector3d(paraso3arry[i])).matrix();
    frame_poses[i + 1].translation() = Eigen::Vector3d(paratarry[i]);
  }
  frame_poses.back() = gnss_pose;
}

void Odom::LoopDetection(std::vector<MapFrame> &history_key_frames, MapFrame &current_frame) {}

} // namespace art
