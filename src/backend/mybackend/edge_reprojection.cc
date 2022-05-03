#include "Thirdparty/Sophus/sophus/se3.hpp"
#include "backend/mybackend/vertex_pose.h"
#include "backend/mybackend/edge_reprojection.h"
#include "utils/utility.h"

#include <iostream>

namespace myslam {
namespace backend {

// 这里面的推导和14讲上的一样，包括正负号，位姿表示为Tcw
/**
 * @brief 三维坐标转归一化平面坐标
 * 
 * @param v 三维坐标
 * @return Vec2 归一化平面坐标
 */
Vec2 project2d(const Vec3& v)  {
    Vec2 res;
    res(0) = v(0)/v(2);
    res(1) = v(1)/v(2);
    return res;
}

/*  
    std::vector<std::shared_ptr<Vertex>> verticies_; // 该边对应的顶点
    VecX residual_;                 // 残差
    std::vector<MatXX> jacobians_;  // 雅可比，每个雅可比维度是 residual x vertex[i]
    MatXX information_;             // 信息矩阵
    VecX observation_;              // 观测信息
*/

void EdgeReprojectionXYZ::ComputeResidual() {

    Vec3 pts_w = verticies_[0]->Parameters();
    VecX pose_i = verticies_[1]->Parameters();              //Tcw
    Qd Qi(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
    Vec3 Pi = pose_i.head<3>();

    Vec3 pts_c = Qi*pts_w + Pi;
    // 误差定义：观测减预测，注意雅可比正负
    residual_ = obs_-cam_project(pts_c);

}

Vec2 EdgeReprojectionXYZ::cam_project(const Vec3 &camera_xyz) const{
    Vec2 proj = project2d(camera_xyz);   //归一化平面坐标
    Vec2 res;                           //像素坐标
    res[0] = proj[0]*fx_ + cx_;
    res[1] = proj[1]*fy_ + cy_;
    return res;
}

bool EdgeReprojectionXYZ::isDepthPositive(){

    Vec3 pts_w = verticies_[0]->Parameters();
    VecX pose_i = verticies_[1]->Parameters();              //Tcw

    Qd Qi(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
    Vec3 Pi = pose_i.head<3>();
    Vec3 pts_c = Qi*pts_w + Pi;

    return pts_c(2)>0.0;
}


void EdgeReprojectionXYZ::ComputeJacobians() {

    Vec3 pts_w = verticies_[0]->Parameters();
    VecX pose_i = verticies_[1]->Parameters();              //Tcw
    Qd Qi(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
    Vec3 Pi = pose_i.head<3>();
    
    Vec3 pts_c = Qi*pts_w + Pi;

    // 相机系下的xy坐标
    double x = pts_c[0];
    double y = pts_c[1];
    double invz = 1.0/pts_c[2];
    double invz_2 = invz*invz;

    Mat23 tmp;
    tmp(0,0) = fx_;
    tmp(0,1) = 0;
    tmp(0,2) = -x*invz*fx_;

    tmp(1,0) = 0;
    tmp(1,1) = fy_;
    tmp(1,2) = -y*invz*fy_;

    // 注意正负，误差计算方式是观测-预测

    Mat23 jacobian_feature;
    jacobian_feature = -1.*invz * tmp * Qi.toRotationMatrix();   //Rcw

    // 这个要看李代数增量是旋转在前还是平移在前
    // 目前是平移在前(Sophus::SE3d)，但是g2o是旋转在前(Sophus::SE3Quat)
    Mat26 jacobian_pose_i;
    jacobian_pose_i(0,0) = -invz *fx_;
    jacobian_pose_i(0,1) = 0;
    jacobian_pose_i(0,2) = x*invz_2 *fx_;
    jacobian_pose_i(0,3) =  x*y*invz_2 *fx_;
    jacobian_pose_i(0,4) = -(1+(x*x*invz_2)) *fx_;
    jacobian_pose_i(0,5) = y*invz *fx_;

    jacobian_pose_i(1,0) = 0;
    jacobian_pose_i(1,1) = -invz *fy_;
    jacobian_pose_i(1,2) = y*invz_2 *fy_;
    jacobian_pose_i(1,3) = (1+y*y*invz_2) *fy_;
    jacobian_pose_i(1,4) = -x*y*invz_2 *fy_;
    jacobian_pose_i(1,5) = -x*invz *fy_;

    // 旋转在前
    // jacobian_pose_i(0,0) =  x*y*invz_2 *fx_;
    // jacobian_pose_i(0,1) = -(1+(x*x*invz_2)) *fx_;
    // jacobian_pose_i(0,2) = y*invz *fx_;
    // jacobian_pose_i(0,3) = -invz *fx_;
    // jacobian_pose_i(0,4) = 0;
    // jacobian_pose_i(0,5) = x*invz_2 *fx_;

    // jacobian_pose_i(1,0) = (1+y*y*invz_2) *fy_;
    // jacobian_pose_i(1,1) = -x*y*invz_2 *fy_;
    // jacobian_pose_i(1,2) = -x*invz *fy_;
    // jacobian_pose_i(1,3) = 0;
    // jacobian_pose_i(1,4) = -invz *fy_;
    // jacobian_pose_i(1,5) = y*invz_2 *fy_;

    jacobians_[0] = jacobian_feature;
    jacobians_[1] = jacobian_pose_i;
}

void EdgeReprojectionPoseOnly::ComputeResidual() {

    VecX pose_i = verticies_[0]->Parameters();              //Tcw
    Qd Qi(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
    Vec3 Pi = pose_i.head<3>();

    Vec3 pts_c = Qi*landmark_world_ + Pi;
    // 误差定义：观测减预测，注意雅可比正负
    residual_ = obs_-cam_project(pts_c);

}

Vec2 EdgeReprojectionPoseOnly::cam_project(const Vec3 &camera_xyz) const{
    Vec2 proj = project2d(camera_xyz);   //归一化平面坐标
    Vec2 res;                           //像素坐标
    res[0] = proj[0]*fx_ + cx_;
    res[1] = proj[1]*fy_ + cy_;
    return res;
}

void EdgeReprojectionPoseOnly::ComputeJacobians() {

    VecX pose_i = verticies_[0]->Parameters();              //Tcw
    Qd Qi(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
    Vec3 Pi = pose_i.head<3>();
    
    Vec3 pts_c = Qi*landmark_world_ + Pi;

    // 相机系下的xy坐标
    double x = pts_c[0];
    double y = pts_c[1];
    double invz = 1.0/pts_c[2];
    double invz_2 = invz*invz;

    // 注意正负，误差计算方式是观测-预测

    // 这个要看李代数增量是旋转在前还是平移在前
    // 目前是平移在前(Sophus::SE3d)，但是g2o是旋转在前(Sophus::SE3Quat)
    Mat26 jacobian_pose_i;
    jacobian_pose_i(0,0) = -invz *fx_;
    jacobian_pose_i(0,1) = 0;
    jacobian_pose_i(0,2) = x*invz_2 *fx_;
    jacobian_pose_i(0,3) =  x*y*invz_2 *fx_;
    jacobian_pose_i(0,4) = -(1+(x*x*invz_2)) *fx_;
    jacobian_pose_i(0,5) = y*invz *fx_;

    jacobian_pose_i(1,0) = 0;
    jacobian_pose_i(1,1) = -invz *fy_;
    jacobian_pose_i(1,2) = y*invz_2 *fy_;
    jacobian_pose_i(1,3) = (1+y*y*invz_2) *fy_;
    jacobian_pose_i(1,4) = -x*y*invz_2 *fy_;
    jacobian_pose_i(1,5) = -x*invz *fy_;

    jacobians_[0] = jacobian_pose_i;

}



// void EdgeReprojection::ComputeResidual() {
// //    std::cout << pts_i_.transpose() <<" "<<pts_j_.transpose()  <<std::endl;

//     double inv_dep_i = verticies_[0]->Parameters()[0];

//     VecX param_i = verticies_[1]->Parameters();
//     Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
//     Vec3 Pi = param_i.head<3>();

//     VecX param_j = verticies_[2]->Parameters();
//     Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
//     Vec3 Pj = param_j.head<3>();

//     VecX param_ext = verticies_[3]->Parameters();
//     Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
//     Vec3 tic = param_ext.head<3>();

//     Vec3 pts_camera_i = pts_i_ / inv_dep_i;
//     Vec3 pts_imu_i = qic * pts_camera_i + tic;
//     Vec3 pts_w = Qi * pts_imu_i + Pi;
//     Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
//     Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

//     double dep_j = pts_camera_j.z();
//     residual_ = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();   /// J^t * J * delta_x = - J^t * r
// //    residual_ = information_ * residual_;   // remove information here, we multi information matrix in problem solver
// }

// //void EdgeReprojection::SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_) {
// //    qic = qic_;
// //    tic = tic_;
// //}

// void EdgeReprojection::ComputeJacobians() {
//     double inv_dep_i = verticies_[0]->Parameters()[0];

//     VecX param_i = verticies_[1]->Parameters();
//     Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
//     Vec3 Pi = param_i.head<3>();

//     VecX param_j = verticies_[2]->Parameters();
//     Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
//     Vec3 Pj = param_j.head<3>();

//     VecX param_ext = verticies_[3]->Parameters();
//     Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
//     Vec3 tic = param_ext.head<3>();

//     Vec3 pts_camera_i = pts_i_ / inv_dep_i;
//     Vec3 pts_imu_i = qic * pts_camera_i + tic;
//     Vec3 pts_w = Qi * pts_imu_i + Pi;
//     Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
//     Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

//     double dep_j = pts_camera_j.z();

//     Mat33 Ri = Qi.toRotationMatrix();
//     Mat33 Rj = Qj.toRotationMatrix();
//     Mat33 ric = qic.toRotationMatrix();
//     Mat23 reduce(2, 3);
//     reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
//         0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
// //    reduce = information_ * reduce;

//     Eigen::Matrix<double, 2, 6> jacobian_pose_i;
//     Eigen::Matrix<double, 3, 6> jaco_i;
//     jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
//     jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Sophus::SO3d::hat(pts_imu_i);
//     jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

//     Eigen::Matrix<double, 2, 6> jacobian_pose_j;
//     Eigen::Matrix<double, 3, 6> jaco_j;
//     jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
//     jaco_j.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_j);
//     jacobian_pose_j.leftCols<6>() = reduce * jaco_j;

//     Eigen::Vector2d jacobian_feature;
//     jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_ * -1.0 / (inv_dep_i * inv_dep_i);

//     Eigen::Matrix<double, 2, 6> jacobian_ex_pose;
//     Eigen::Matrix<double, 3, 6> jaco_ex;
//     jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
//     Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
//     jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
//                              Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
//     jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;

//     jacobians_[0] = jacobian_feature;
//     jacobians_[1] = jacobian_pose_i;
//     jacobians_[2] = jacobian_pose_j;
//     jacobians_[3] = jacobian_ex_pose;

//     ///------------- check jacobians -----------------
// //    {
// //        std::cout << jacobians_[0] <<std::endl;
// //        const double eps = 1e-6;
// //        inv_dep_i += eps;
// //        Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
// //        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
// //        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
// //        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
// //        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
// //
// //        Eigen::Vector2d tmp_residual;
// //        double dep_j = pts_camera_j.z();
// //        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();
// //        tmp_residual = information_ * tmp_residual;
// //        std::cout <<"num jacobian: "<<  (tmp_residual - residual_) / eps <<std::endl;
// //    }

// }

}
}