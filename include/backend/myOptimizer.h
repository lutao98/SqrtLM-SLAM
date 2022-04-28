#ifndef MYOPTIMIZER_H
#define MYOPTIMIZER_H

#include "data_structure/Map.h"
#include "data_structure/MapPoint.h"
#include "data_structure/KeyFrame.h"
#include "data_structure/Frame.h"
#include "LoopClosing.h"

#include "backend/mybackend/backend.h"
#include "backend/mybackend/vertex_point_xyz.h"
#include "backend/mybackend/vertex_pose.h"
#include "backend/mybackend/edge_reprojection.h"
#include "backend/mybackend/loss_function.h"
#include "backend/mybackend/problem.h"
namespace ORB_SLAM2
{

class LoopClosing;

class MyOptimizer
{
public:
    
    void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);

    int static PoseOptimization(Frame* pFrame);

};

} //namespace ORB_SLAM

#endif // CERESOPTIMIZER_H