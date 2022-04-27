#include "backend/mybackend/vertex_pose.h"
#include "Thirdparty/Sophus/sophus/se3.hpp"
//#include <iostream>
namespace myslam {
namespace backend {

void VertexPose::Plus(const VecX &delta) {
    VecX &parameters = Parameters();      //注意是Tcw
    parameters.head<3>() += delta.head<3>();     //李代数增量表示是平移在前，旋转在后，g2o是旋转在前，平移在后
    Qd q(parameters[6], parameters[3], parameters[4], parameters[5]);
    // 左乘更新
    q = Sophus::SO3d::exp(Vec3(delta[3], delta[4], delta[5])).unit_quaternion() * q;  // left multiplication with so3
    // q = q * Sophus::SO3d::exp(Vec3(delta[3], delta[4], delta[5])).unit_quaternion();  // right multiplication with so3
    q.normalized();
    parameters[3] = q.x();
    parameters[4] = q.y();
    parameters[5] = q.z();
    parameters[6] = q.w();
//    Qd test = Sophus::SO3d::exp(Vec3(0.2, 0.1, 0.1)).unit_quaternion() * Sophus::SO3d::exp(-Vec3(0.2, 0.1, 0.1)).unit_quaternion();
//    std::cout << test.x()<<" "<< test.y()<<" "<<test.z()<<" "<<test.w() <<std::endl;
}

}
}
