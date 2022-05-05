#include "backend/mybackend/vertex_pose.h"
#include "Thirdparty/Sophus/sophus/se3.hpp"
#include <iostream>
namespace myslam {
namespace backend {

void VertexPose::Plus(const VecX &delta) {
    VecX &parameters = Parameters();      //注意是Tcw

    Sophus::SE3d pose(Qd(parameters[6], parameters[3], parameters[4], parameters[5]),parameters.head<3>());
    // 对于Sophus::SE3d,需要的李代数增量为平移在前
    Sophus::SE3d update=Sophus::SE3d::exp(delta)*pose;   // 注意这里的数学含义
    parameters.tail<4>() = update.unit_quaternion().coeffs();
    parameters.head<3>() = update.translation();

    // 以下更新方式是错的
    // parameters.head<3>() += delta.head<3>();     //李代数增量表示是平移在前，旋转在后，g2o是旋转在前，平移在后
    // Qd q(parameters[6], parameters[3], parameters[4], parameters[5]);
    // // 左乘更新
    // q = Sophus::SO3d::exp(Vec3(delta.tail<3>())).unit_quaternion() * q;  // left multiplication with so3
    // q.normalized();
    
}

}
}
