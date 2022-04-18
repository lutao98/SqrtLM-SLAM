// created by lutao
// 2021.7.4
#ifndef TYPES_LIDAR_H
#define TYPES_LIDAR_H

#include "../core/base_vertex.h"
#include "../core/base_binary_edge.h"
#include "../core/base_unary_edge.h"
#include "se3_ops.h"
#include "se3quat.h"
#include "types_sba.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {

class VertexLidarFlatPoint : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
  VertexLidarFlatPoint() {}
};

}// end namespace

#endif // TYPES_LIDAR_H
