#ifndef MYSLAM_BACKEND_POSEVERTEX_H
#define MYSLAM_BACKEND_POSEVERTEX_H

#include <memory>
#include "vertex.h"

namespace myslam {
namespace backend {

/**
 * Pose vertex
 * parameters: tx, ty, tz, qx, qy, qz, qw, 7 DoF，
 * g2o是旋转在前，平移在后
 * optimization is perform on manifold, so update is 6 DoF, left multiplication
 *
 * pose is represented as ！！！！Tcw！！！！ in ORB-SLAM case
 */
class VertexPose : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPose() : Vertex(7, 6) {}

    /// 加法，可重定义
    /// 注意是Tcw，t向量加，Q左乘更新
    virtual void Plus(const VecX &delta) override;

    std::string TypeInfo() const {
        return "VertexPose";
    }

};

}
}

#endif
