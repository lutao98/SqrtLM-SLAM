/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "backend/Optimizer.h"


namespace ORB_SLAM2
{
    Optimizer::eSolver solver=Optimizer::G2O;
    // Optimizer::eSolver solver=Optimizer::MYOPT;
    bool is_use_ceres=false;

void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    if(is_use_ceres)
        CeresOptimizer::GlobalBundleAdjustemnt(pMap, nIterations, pbStopFlag, nLoopKF,bRobust);
    else
        g2oOptimizer::GlobalBundleAdjustemnt(pMap, nIterations, pbStopFlag, nLoopKF,bRobust);
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    if(is_use_ceres)
        CeresOptimizer::BundleAdjustment(vpKFs,vpMP,nIterations, pbStopFlag, nLoopKF, bRobust);
    else
        g2oOptimizer::BundleAdjustment(vpKFs,vpMP,nIterations, pbStopFlag, nLoopKF, bRobust);

}

int Optimizer::PoseOptimization(Frame *pFrame, PointICloudPtr local_lidarmap_cloud_ptr,
                                pcl::KdTreeFLANN<PointI>::Ptr kdtree_local_map, const lidarConfig* lidarconfig)
{
    TicToc op_tic;
    int inlier_num=0;
    if(solver==Optimizer::CERES)
        inlier_num = CeresOptimizer::PoseOptimization(pFrame);
    else if(solver==Optimizer::G2O)
        inlier_num = g2oOptimizer::PoseOptimization(pFrame, local_lidarmap_cloud_ptr, kdtree_local_map, lidarconfig);
    else
        inlier_num = MyOptimizer::PoseOptimization(pFrame);
    std::cout << "             ::PoseOptimization() 耗时" << op_tic.toc() << " ms." << std::endl;
    return inlier_num;
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, const lidarConfig* lidarconfig)
{
    TicToc op_tic;
    if(is_use_ceres)
        CeresOptimizer::LocalBundleAdjustment(pKF,pbStopFlag,pMap);
    else
        g2oOptimizer::LocalBundleAdjustment(pKF,pbStopFlag,pMap,lidarconfig);
    std::cout << "[LocalMapping](4)::LocalBundleAdjustment() 耗时" << op_tic.toc() << " ms." << std::endl;
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    if(is_use_ceres)
        CeresOptimizer::OptimizeEssentialGraph(pMap,pLoopKF, pCurKF,
                                               NonCorrectedSim3,
                                               CorrectedSim3,
                                               LoopConnections, bFixScale);
    else
        g2oOptimizer::OptimizeEssentialGraph(pMap,pLoopKF, pCurKF,
                                             NonCorrectedSim3,
                                             CorrectedSim3,
                                             LoopConnections, bFixScale);

}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    if(is_use_ceres)
        CeresOptimizer::OptimizeSim3(pKF1, pKF2, vpMatches1, g2oS12, th2, bFixScale);
    else
        g2oOptimizer::OptimizeSim3(pKF1, pKF2, vpMatches1, g2oS12, th2, bFixScale);
    return 0;
}


} //namespace ORB_SLAM
