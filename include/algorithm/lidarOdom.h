#define PCL_NO_PRECOMPILE
#ifndef ART_ODOM_H_
#define ART_ODOM_H_
#include <cmath>
#include <deque>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/pcl_macros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils/lidarFactor.h"
#include "utils/CircularBuffer.h"
#include "data_structure/point_types.h"

// 单独跑激光可以用到
namespace art {
using namespace std;
using namespace lom;
using namespace mathutils;

typedef pcl::PointXYZI PointI;
typedef typename pcl::PointCloud<PointI> PointICloud;
typedef typename pcl::PointCloud<PointI>::Ptr PointICloudPtr;
typedef typename pcl::PointCloud<PointI>::ConstPtr PointICloudConstPtr;

typedef PointXYZIR PointIR;
typedef typename pcl::PointCloud<PointIR> PointIRCloud;
typedef typename pcl::PointCloud<PointIR>::Ptr PointIRCloudPtr;
typedef typename pcl::PointCloud<PointIR>::ConstPtr PointIRCloudConstPtr;

typedef PointXYZIRT PointIRT;
typedef typename pcl::PointCloud<PointIRT> PointIRTCloud;
typedef typename pcl::PointCloud<PointIRT>::Ptr PointIRTCloudPtr;
typedef typename pcl::PointCloud<PointIRT>::ConstPtr PointIRTCloudConstPtr;

typedef std::pair<size_t, size_t> IndexRange;

struct ImageElement {
  int point_state = -3;   // -3=没有占用  -1=地面点  -2=聚类不稳定的类  0=非地面点
  int feature_state = 0;   // 1=角点/平面点
  int index = 0;
};

struct ImageIndex{
  int row = 0;
  int col = 0;
};

struct OdomConfig {
  bool using_livox = false;
  bool using_rs80 = false;
  bool using_32c = false;
  bool even = true;

  int sensor_type = 64;
  double deg_diff = 0.08;
  double scan_period = 0.1;
  double lower_bound = -24.9;
  double upper_bound = 2;
  double ground_z_bound = -1.7;             // 地面高度   只会影响角点的提取
  int using_lower_ring_num = 1;
  int using_upper_ring_num = 64;
  int lower_ring_num_sharp_point = 1;      // 角点范围
  int upper_ring_num_sharp_point = 64;      // 角点范围
  int lower_ring_num_z_trans = 1;           // 估计z平移的平面点范围
  int upper_ring_num_z_trans = 64;          // 估计z平移的平面点范围
  int lower_ring_num_x_rot = 1;             // 估计x旋转的平面点范围
  int upper_ring_num_x_rot = 64;            // 估计x旋转的平面点范围
  int lower_ring_num_y_rot = 1;             // 估计y旋转的平面点范围
  int upper_ring_num_y_rot = 64;            // 估计y旋转的平面点范围
  int lower_ring_num_z_rot_xy_trans = 1;   // 估计z旋转和xy平移的平面点范围
  int upper_ring_num_z_rot_xy_trans = 64;   // 估计z旋转和xy平移的平面点范围
  int num_scan_subregions = 8;              // 一线分为多少个子区域
  int num_curvature_regions_corner = 5;     // 算角点时候的邻域
  int num_curvature_regions_flat = 5;       // 算平面点时候的邻域
  int num_feature_regions = 5;
  double surf_curv_th = 0.01;               // 平面点曲率阈值  越小越平
  double sharp_curv_th = 10;                 // 角点曲率阈值  越大越好
  int max_corner_sharp = 4;                 // 前max_corner_sharp个曲率大的点被设置为角点
  int max_corner_less_sharp = 8;            // 前max_corner_less_sharp个曲率大的点被设置为弱角点
  int max_surf_flat = 4;
  int max_surf_less_flat = 8;
  double max_sq_dis = 1;                    // 提取平面点时的邻域大小
  int flat_extract_num_x_trans = 150;       // 使用这部分点的数量
  int flat_extract_num_y_trans = 150;
  int flat_extract_num_z_trans = 150;
  int flat_extract_num_x_rot = 150;
  int flat_extract_num_y_rot = 150;
  int flat_extract_num_z_rot = 150;
  double less_flat_filter_size = 0.2;
  bool using_corner_point_vector = false;
  bool using_surf_point_normal = false;
  bool using_sharp_point = true;
  bool using_flat_point = true;
  double distance_sq_threshold = 0.2;       // 匹配时候，搜索特帧点最近邻点的距离阈值，必须要小于这个阈值才认为是同一个点
  void SetupParam(ros::NodeHandle &nh);
  void setParam(cv::FileStorage &cfg);
};

struct MapFrame {
  PointIRTCloud all_points;
  PointIRTCloud feature_points;
  Eigen::Affine3d transform{Eigen::Affine3d::Identity()};
  bool gnssStatus{true};
  MapFrame(){};
};

class Odom {

 public:

  Odom() = delete;

  Odom(const OdomConfig &config, const ros::NodeHandle &nh);

  void EstimatePoseForSLAM(const PointIRTCloud &all_cloud_in, Eigen::Affine3d &estimate_pose);

  void EstimatePoseForSLAM(const PointIRTCloud &all_cloud_in, Eigen::Affine3d &estimate_pose, Eigen::Affine3d &transf_last_curr);

  void EstimatePoseForSLAM(const PointIRTCloud &all_cloud_in, PointIRTCloud &feature_cloud, Eigen::Affine3d &estimate_pose);

  void EstimatePoseForMapping(const PointIRTCloud &all_cloud_in, PointIRTCloud &feature_cloud, Eigen::Affine3d &estimate_pose, Eigen::Affine3d &transf_last_curr);

  void EstimatePoseForMapping(const std::deque<MapFrame> &local_map_frames, std::deque<MapFrame> &estimate_frames, Eigen::Affine3d &transf_last_curr);

  void EstimatePoseForLocalization(const PointIRTCloud &source_cloud, Eigen::Affine3d& estimate_pose, Eigen::Affine3d& transf_last_curr);

  void setInputTarget(const PointIRTCloud &target_cloud);

  void SystemInital();

  void ExtractFeaturePoints();

  void GetFeaturePoints(const PointIRTCloud &source_cloud, PointIRTCloud &feature_cloud);

  void PublishResults();

  void FrontEndForSLAM();

  void FrontEndForMapping();

  void FrontEndForMapping(const std::deque<MapFrame> &local_map_frames);

  void FrontEndForLocalization();

  void BackEndForLoop(std::vector<Eigen::Affine3d> &frame_poses, const Eigen::Affine3d &loop_pose);

  void BackEndForGNSS(std::vector<Eigen::Affine3d> &frame_poses, const Eigen::Affine3d &gnss_pose);

  void LoopDetection(std::vector<MapFrame> &history_key_frames, MapFrame &current_frame);

  void TransformToEnd(const PointIRT &pi, PointIRT &po);

  void RotateVectToEnd(const PointIRT &pi, PointIRT &po);

  void pubFeaturePoint();

  vector<PointIRTCloudPtr> all_laser_scans_;

  vector<PointIRTCloudPtr> laser_scans_;

  // 每行范围
  vector<IndexRange> scan_ranges_;

  vector<vector<ImageElement>> range_image_;

  vector<vector<ImageIndex>> image_index_;

  vector<ImageIndex> surface_points_less_flat_index_;

  vector<vector<float>> ringAng;

  vector<float> scanAng;

 protected:

  ros::Time sweep_start_;
  ros::Time scan_time_;

  int row_num_;
  int col_num_;

  double factor_;
  float time_factor_;

  OdomConfig config_;

  cv::Mat range_img_visual_;

  PointIRTCloud cloud_in_rings_;
  PointIRTCloud corner_points_sharp_;
  PointIRTCloud corner_points_less_sharp_;
  PointIRTCloud surface_points_flat_;
  PointIRTCloud surface_points_less_flat_;
  PointIRTCloud surface_points_flat_z_trans_;
  PointIRTCloud surface_points_flat_z_rot_xy_trans_;
  PointIRTCloud surface_points_flat_x_rot_;
  PointIRTCloud surface_points_flat_y_rot_;
  PointIRTCloud mask_points_;


  PointIRTCloud corner_points_vector_;
  PointIRTCloud surface_points_flat_normal_;
  PointIRTCloud surface_points_less_flat_normal_;

  PointIRTCloud corner_points_sharp_last_;
  PointIRTCloud surface_points_flat_last_;
  PointIRTCloud corner_points_less_sharp_last_;
  PointIRTCloud surface_points_less_flat_last_;

  PointIRTCloudPtr map_cloud_ptr_;
  PointIRTCloud all_cloud_;

  vector<vector<ImageIndex>> less_sharp_;
  vector<ImageIndex> sharp_;

  void CalculateRingAndTime(const PointICloud &all_cloud_in);
  void CalculateRingAndTime(const PointIRCloud &all_cloud_in);
  void CalculateRingAndTime(const PointIRTCloud &all_cloud_in);
  void PointToImage(const PointIRTCloud &all_cloud_in);

  vector<int> scan_ring_mask_;
  vector<pair<float, size_t> > curvature_idx_pairs_; // in subregion

  void Reset();
  void PrepareRing_corner(const PointIRTCloud &scan);
  void PrepareRing_corner(const vector<PointIRTCloudPtr> &scan);
  void PrepareRing_flat(const PointIRTCloud &scan);
  void PrepareSubregion_corner(const PointIRTCloud &scan, const size_t idx_start, const size_t idx_end);
  void PrepareSubregion_flat(const PointIRTCloud &scan, const size_t idx_start, const size_t idx_end);
  void MaskPickedInRing(const PointIRTCloud &scan, const size_t in_scan_idx);

  bool system_inited_;

  CircularBuffer<PointIRTCloud> local_map_;
  CircularBuffer<Eigen::Affine3d> trans_sum_for_local_map_;

  PointIRTCloudPtr local_map_cloud_ptr_;

  pcl::KdTreeFLANN<PointIRT>::Ptr kdtree_map_;
  pcl::KdTreeFLANN<PointIRT>::Ptr kdtree_local_map_;  

  double para_q_[4] = {0, 0, 0, 0};
  double para_so3_[3] = {0, 0, 0};
  double para_t_[3] = {0, 0, 0};

  Eigen::Affine3d estimate_pose_;
  Eigen::Affine3d transf_last_curr_;

 private:
  float start_ori_, end_ori_;
  ros::NodeHandle nh_;
  ros::Publisher pub_processed_cloud_;
  ros::Publisher pub_sharp_cloud_;
  ros::Publisher pub_flat_cloud_;
  ros::Publisher pub_less_sharp_cloud_;
  ros::Publisher pub_less_flat_cloud_;
  ros::Publisher pub_rangeimage_;
  ros::Publisher pub_localmap_;
};


void CalculateRingAndTime(const PointICloud &all_cloud_in, PointIRTCloud &all_cloud_out) {

  //激光雷达线数初始化为64
  int N_SCANS = 64;
  //扫描周期, velodyne频率10Hz，周期0.1s
  const double scanPeriod = 0.1;

  size_t point_size = all_cloud_in.size();

  //每次扫描是一条线，看作者的数据集激光x向前，y向左，那么下面就是线一端到另一端
  //atan2的输出为-pi到pi(PS:atan输出为-pi/2到pi/2)
  //计算旋转角时取负号是因为velodyne是顺时针旋转
  float startOri = -atan2(all_cloud_in.points[0].y, all_cloud_in.points[0].x)+ M_PI;
  float endOri = -atan2(all_cloud_in.points[point_size - 1].y,
                        all_cloud_in.points[point_size - 1].x) + 3 * M_PI;

  //激光间距收束到1pi到3pi
  if (endOri - startOri > 3 * M_PI)
  {
      endOri -= 2 * M_PI;
  }
  else if (endOri - startOri < M_PI)
  {
      endOri += 2 * M_PI;
  }
  printf("start Ori %fPI\nend Ori %fPI\n", startOri/M_PI, endOri/M_PI);

  //记录总点数
  int count = point_size;

  //按线数保存的点云集合
  std::vector<pcl::PointCloud<PointIRT>> laserCloudScans(N_SCANS);

  PointIRT p;
  double max_angle=-10000, min_angle=10000;
  for (size_t i = 0; i < point_size; ++i) {
    p.x = all_cloud_in.points[i].x;
    p.y = all_cloud_in.points[i].y;
    p.z = all_cloud_in.points[i].z;
    p.intensity = all_cloud_in.points[i].intensity;
    p.timestamp = 0.1;

    //求仰角atan输出为-pi/2到pi/2，实际看scanID应该每条线之间差距是2度
    float angle = atan(p.z / sqrt(p.x * p.x + p.y * p.y)) * 180 / M_PI;
    int scanID = -1;
    if(angle<min_angle) min_angle=angle;
    if(angle>max_angle) max_angle=angle;


    if (angle >= -8.83)
        scanID = int((2 - angle) * 3.0 + 0.5);
    else
        scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);
//        printf("angle %f scanID %d \n", angle, scanID);

    if (scanID > (N_SCANS - 1) || scanID < 0)
    {
      count--;
      continue;
    }

    p.ring = scanID;
    //按线分类保存
    laserCloudScans[scanID].push_back(p);
    all_cloud_out.points.push_back(p);
  }

  point_size = count;
  std::cout << "[CalculateRingAndTime]::总点数:" << all_cloud_in.points.size()
            << "       虚拟线束化点数:" << point_size
            << "    max angle:" << max_angle
            << "    min angle:" << min_angle << std::endl;
}


void CalculateRingAndTime(const PointIRCloud &all_cloud_in, PointIRTCloud &all_cloud_out) {
  PointIRT p;
  size_t point_size = all_cloud_in.size();

  float startOri = -atan2(all_cloud_in.points[0].y, all_cloud_in.points[0].x);
  float endOri = -atan2(all_cloud_in.points[point_size - 1].y,
                        all_cloud_in.points[point_size - 1].x) + 2 * M_PI;
   if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

  float first_point_azi_offset = 0.0;
  switch (all_cloud_in.points[0].ring) {
    case 0: case 15:
    case 2: case 4:
    case 19: case 7:
    case 23: case 26:
    case 11: case 28: case 30:
      first_point_azi_offset = 1.4;
      break;

    case 17:
    case 21: case 25:
    case 9: case 13:
      first_point_azi_offset = -4.2;
      break;

    case 1: case 20:
    case 3: case 5:
    case 24: case 8:
    case 27: case 12:
    case 29: case 31: case 16:
      first_point_azi_offset = -1.4;
      break;

    case 18: case 6:
    case 22: case 10: case 14:
      first_point_azi_offset = 4.2;
      break;

    default:
      break;
  }

  bool halfPassed = false;
  for (size_t i = 0; i < point_size; ++i) {
    p.x = all_cloud_in.points[i].x;
    p.y = all_cloud_in.points[i].y;
    p.z = all_cloud_in.points[i].z;
    p.intensity = all_cloud_in.points[i].intensity;
    p.ring = all_cloud_in.points[i].ring;

    float azi_rad = -atan2(p.y, p.x);
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

    float azi_offset = 0.0;
    switch (p.ring) {
      case 0: case 15:
      case 2: case 4:
      case 19: case 7:
      case 23: case 26:
      case 11: case 28: case 30:
        azi_offset = first_point_azi_offset - 1.4;
        break;

      case 17:
      case 21: case 25:
      case 9: case 13:
        azi_offset = first_point_azi_offset + 4.2;
        break;

      case 1: case 20:
      case 3: case 5:
      case 24: case 8:
      case 27: case 12:
      case 29: case 31: case 16:
        azi_offset = first_point_azi_offset + 1.4;
        break;

      case 18: case 6:
      case 22: case 10: case 14:
        azi_offset = first_point_azi_offset - 4.2;
        break;

      default:
        break;
    }

    float azi_rad_rel = azi_rad - startOri + azi_offset / 180.0 * M_PI;
    p.timestamp = azi_rad_rel / (2 * M_PI) * 0.1;
    if (p.timestamp < 0.0) {
      p.timestamp = 0.0;
    }
    all_cloud_out.points.push_back(p);
  }
  std::cout << "[CalculateRingAndTime]::32c   input size:" << all_cloud_in.size()
            << "output size:" << all_cloud_out.size() << std::endl;
}


void CalculateRingAndTime(const PointIRTCloud &all_cloud_in, PointIRTCloud &all_cloud_out) {
  PointIRT p;
  size_t point_size = all_cloud_in.size();
  for (size_t i = 0; i < point_size; ++i) {
    p.x = all_cloud_in.points[i].x;
    p.y = all_cloud_in.points[i].y;
    p.z = all_cloud_in.points[i].z;
    p.intensity = all_cloud_in.points[i].intensity;
    p.ring = all_cloud_in.points[i].ring;
    p.timestamp = (all_cloud_in.points[i].timestamp - all_cloud_in.points[0].timestamp)
                / (all_cloud_in.points[point_size - 1].timestamp - all_cloud_in.points[0].timestamp) * 0.1;
    if (p.timestamp < 0.0) {
      p.timestamp = 0.0;
    }
    all_cloud_out.points.push_back(p);
  }
}
} // namespace art
#endif
