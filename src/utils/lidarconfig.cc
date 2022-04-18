#include "utils/lidarconfig.h"

void lidarConfig::setParam(cv::FileStorage &cfg){
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

  corner_optimized_weight=cfg["corner_optimized_weight"];
  flat_optimized_weight=cfg["flat_optimized_weight"];
}
