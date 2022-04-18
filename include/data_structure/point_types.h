#ifndef LOM_POINT_TYPES_H_
#define LOM_POINT_TYPES_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

/** Euclidean coordinate, including intensity and ring number. */
struct PointXYZIR
{
  PCL_ADD_POINT4D;                // quad-word XYZ
  PCL_ADD_INTENSITY;
  uint16_t ring;                ///< laser ring number
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ensure proper alignment
}EIGEN_ALIGN16;

struct PointXYZIRT
{
  PCL_ADD_POINT4D;                // quad-word XYZ
  uint8_t intensity;
  uint16_t ring;                ///< laser ring number
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ensure proper alignment
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring)
                                  )

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (uint8_t, intensity, intensity)
                                  (uint16_t, ring, ring)
                                  (double, timestamp, timestamp)
                                  )

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

#endif  // LOM_POINT_TYPES_H_
