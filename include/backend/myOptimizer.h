#ifndef MYOPTIMIZER_H
#define MYOPTIMIZER_H

#include "data_structure/Map.h"
#include "data_structure/MapPoint.h"
#include "data_structure/KeyFrame.h"
#include "data_structure/Frame.h"
#include "LoopClosing.h"

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