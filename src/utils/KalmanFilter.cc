#include "utils/KalmanFilter.h"

namespace kalman_filter
{
  CKalmanFilter::CKalmanFilter(){};
  CKalmanFilter::~CKalmanFilter(){};
  /* 
   * 初始化卡尔曼滤波类
   * 输入：状态估计误差的协方差矩阵对角元Rt1和Rt2
   *       测量误差的协方差矩阵对角元Qt1和Qt2
   *       两次测量时间差dt
   * 输出：无
   * 返回：无
   */
  void CKalmanFilter::InitClass() {
    a_t.setIdentity();
    a_t.topRightCorner<3, 3>().setIdentity();
    c_t.setIdentity();
    r_t.setIdentity();
    q_t.setIdentity();
    r_t.topLeftCorner<3, 3>() = 0.01 * 0.01 * Matrix3d().setIdentity();
    r_t.bottomRightCorner<3, 3>() = 0.05 * 0.05 * Matrix3d().setIdentity();
    q_t.topLeftCorner<3, 3>() = 0.2 * 0.2 * Matrix3d().setIdentity();
    q_t.bottomRightCorner<3, 3>() = 0.2 * 0.2 * Matrix3d().setIdentity();
  };

  void CKalmanFilter::PositionFilter(const Matrix<double, 6, 1> obsv,
                                    Matrix<double, 6, 1>& state_out)
  {
    // static variable
    static Matrix<double, 6, 1> miu_t = obsv;
    static Matrix<double, 6, 6> sigma_t = Matrix<double, 6, 6>().setIdentity();
    Matrix<double, 6, 6> eye_mat = Matrix<double, 6, 6>().setIdentity();
    Matrix<double, 6, 1> z_t = obsv;
    Matrix<double, 6, 1> miu_t_hat = a_t * miu_t;
    Matrix<double, 6, 6> sigma_t_hat = a_t * sigma_t * a_t.transpose() + r_t;
    Matrix<double, 6, 6> temp_mat = c_t * sigma_t_hat * c_t.transpose() + q_t;
    Matrix<double, 6, 6> k_t = sigma_t_hat * c_t.transpose() * temp_mat.inverse();
    miu_t = miu_t_hat + k_t * (z_t - c_t * miu_t_hat);
    state_out = miu_t;
    sigma_t = (eye_mat - k_t * c_t) * sigma_t_hat;
  }
} //end namespace
