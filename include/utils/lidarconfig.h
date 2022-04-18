#ifndef LIDARCONFIG_H
#define LIDARCONFIG_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

struct lidarConfig {
  bool using_livox = false;
  bool using_rs80 = false;
  bool using_32c = false;
  bool even = true;

  int sensor_type = 64;
  double deg_diff = 0.2;
  double scan_period = 0.1;
  double lower_bound = -24.9;
  double upper_bound = 2;
  double ground_z_bound = -1.5;             // 地面高度   只会影响角点的提取
  int using_lower_ring_num = 1;
  int using_upper_ring_num = 64;
  int lower_ring_num_sharp_point = 1;      // 角点范围
  int upper_ring_num_sharp_point = 40;      // 角点范围
  int lower_ring_num_z_trans = 5;           // 估计z平移的平面点范围
  int upper_ring_num_z_trans = 50;          // 估计z平移的平面点范围
  int lower_ring_num_x_rot = 5;             // 估计x旋转的平面点范围
  int upper_ring_num_x_rot = 50;            // 估计x旋转的平面点范围
  int lower_ring_num_y_rot = 5;             // 估计y旋转的平面点范围
  int upper_ring_num_y_rot = 50;            // 估计y旋转的平面点范围
  int lower_ring_num_z_rot_xy_trans = 5;   // 估计z旋转和xy平移的平面点范围
  int upper_ring_num_z_rot_xy_trans = 50;   // 估计z旋转和xy平移的平面点范围
  int num_scan_subregions = 8;              // 一线分为多少个子区域
  int num_curvature_regions_corner = 3;     // 算角点时候的邻域
  int num_curvature_regions_flat = 5;       // 算平面点时候的邻域
  int num_feature_regions = 5;
  double surf_curv_th = 0.001;               // 平面点曲率阈值  越小越平
  double sharp_curv_th = 20;                 // 角点曲率阈值  越大越好
  int max_corner_sharp = 3;                 // 前max_corner_sharp个曲率大的点被设置为角点
  int max_corner_less_sharp = 8;            // 前max_corner_less_sharp个曲率大的点被设置为弱角点
  int max_surf_flat = 2;
  int max_surf_less_flat = 6;
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

  double corner_optimized_weight=30;
  double flat_optimized_weight=50;

  int row_num_= sensor_type, col_num_= 360.0/deg_diff;
  double factor_=((sensor_type -1) / (upper_bound - lower_bound));
  float time_factor_=1/scan_period;

  void setParam(cv::FileStorage &cfg);
};
#endif // LIDARCONFIG_H
