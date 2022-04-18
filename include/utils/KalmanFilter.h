#ifndef C_KALMAN_FILTER_H
#define C_KALMAN_FILTER_H
#include <glog/logging.h>
#include <eigen3/Eigen/Dense>
namespace kalman_filter
{
using namespace Eigen;
class CKalmanFilter
{
public:
  CKalmanFilter();      //默认构造函数
  ~CKalmanFilter();     //析构函数
  /*
   * 初始化卡尔曼滤波类
   * 输入：状态估计误差的协方差矩阵对角元Rt1和Rt2
   *       测量误差的协方差矩阵对角元Qt1和Qt2
   *       两次测量时间差dt
   * 输出：无
   * 返回：无
   */
  void InitClass();

  void PositionFilter(const Matrix<double, 6, 1> obsv,
                     Matrix<double, 6, 1>& state_out);
private:
  Matrix<double, 6, 6> a_t;
  Matrix<double, 6, 6> c_t;
  Matrix<double, 6, 6> r_t;   // 状态估计误差的协方差矩阵（设为对角）
  Matrix<double, 6, 6> q_t;   // 测量误差的协方差矩阵（设为对角）
}; // end CKalmanFilter
}  // end namespace

#endif