/**
 * @file myOptimizer.cc
 * @author lutao 
 * @brief 手写优化器
 * @version 0.1
 * @date 2022-4-27
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "backend/myOptimizer.h"
#include "utils/Converter.h"

using namespace myslam;

namespace ORB_SLAM2
{

/**
 * @brief Pose Only Optimization
 * 
 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw)
 * 只优化Frame的Tcw，不优化MapPoints的坐标
 * 
 * 1. Vertex: backend::VertexPose()，即当前帧的Tcw
 * 2. Edge:
 *     - g2o::EdgeReprojectionPoseOnly()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的像素坐标(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *
 * @param   pFrame Frame
 * @return  inliers数量
 */
int MyOptimizer::PoseOptimization(Frame *pFrame)
{
    // 该优化函数主要用于Tracking线程中：运动跟踪、参考帧跟踪、地图跟踪、重定位

    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
    const double deltaMono = sqrt(5.991);
    // 自由度为3的卡方分布，显著性水平为0.05，对应的临界阈值7.815   
    const double deltaStereo = sqrt(7.815);

    // 核函数
    backend::LossFunction *lossfunction;
    lossfunction = new backend::CauchyLoss(deltaMono);
    //    lossfunction = new backend::TukeyLoss(1.0);

    // step1. 构建 problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    problem.setDebugOutput(false);       //调试输出:迭代 耗时等信息

    // 输入的帧中,有效的,参与优化过程的2D-3D点对
    int nInitialCorrespondences=0;

    // Set Frame vertex
    // Step 2：添加顶点：待优化当前帧的Tcw
    shared_ptr<backend::VertexPose> vSE3(new backend::VertexPose());
    Qd q;
    Vec3 t;
    Converter::toEigenQT(pFrame->mTcw,q,t);
    Vec7 pose;
    pose.head<3>() = t;
    pose.tail<4>() = Vec4(q.coeffs());// q的初始化顺序wxyz，实际存储顺序xyzw
    vSE3->SetParameters(pose);        // parameters: tx, ty, tz, qx, qy, qz, qw, 7 DoF
    vSE3->SetFixed(false);
    problem.AddVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<shared_ptr<backend::EdgeReprojectionPoseOnly>> vpEdges;
    vector<size_t> vnIndexEdge;
    vpEdges.reserve(N);
    vnIndexEdge.reserve(N);

    // Step 3：添加一元边
    {
    // 锁定地图点。由于需要使用地图点来构造顶点和边,因此不希望在构造的过程中部分地图点被改写造成不一致甚至是段错误
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    // 遍历当前地图中的所有地图点
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        // 如果这个地图点还存在没有被剔除掉
        if(pMP)
        {

            nInitialCorrespondences++;
            pFrame->mvbOutlier[i] = false;

            // 对这个地图点的观测
            Vec2 obs;
            const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
            obs << kpUn.pt.x, kpUn.pt.y;
            // 新建节点,注意这个节点的只是优化位姿Pose
            shared_ptr<backend::EdgeReprojectionPoseOnly> e(new backend::EdgeReprojectionPoseOnly(obs));
            // 填充
            e->AddVertex(vSE3);
            // 这个点的可信程度和特征点所在的图层有关
            const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
            e->SetInformation(Mat22::Identity()*invSigma2);
            // 在这里使用了鲁棒核函数
            e->SetLossFunction(lossfunction);

            // 设置相机内参
            e->setCamIntrinsics(pFrame->fx,pFrame->fy,pFrame->cx,pFrame->cy);

            // 地图点的世界坐标,根据vertex(pose)投影算预测值
            cv::Mat Pw = pMP->GetWorldPos();
            e->setLandmarkWorld(Vec3(Pw.at<float>(0),Pw.at<float>(1),Pw.at<float>(2)));

            problem.AddEdge(e);

            vpEdges.push_back(e);
            vnIndexEdge.push_back(i);

        }

    }
    } // 离开临界区

    // 如果没有足够的匹配点,那么就只好放弃了
    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // Step 4：开始优化，总共优化四次，每次优化迭代10次,每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};          // 单目
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};       // 双目
    const int its[4]={10,10,10,10};// 四次迭代，每次迭代的次数

    // bad 的地图点个数
    int nBad=0;
    // 一共进行四次优化
    for(size_t it=0; it<4; it++)
    {

        // 感觉这句话没必要加 每次迭代后应该更新顶点位姿 这里是固定了每次迭代顶点位姿的初始值
        // 感觉也可以加 因为每次迭代后外点被剔除了 优化结果会更好
        vSE3->SetParameters(pose);     // parameters: tx, ty, tz, qx, qy, qz, qw, 7 DoF

        // 其实就是初始化优化器,这里的参数0就算是不填写,默认也是0,也就是只对level为0的边进行优化
        problem.setOptimizeLevel(0);
        // 开始优化，优化10次
        problem.Solve(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdges.size(); i<iend; i++)
        {
            shared_ptr<backend::EdgeReprojectionPoseOnly> e = vpEdges[i];

            const size_t idx = vnIndexEdge[i];

            // 如果这条误差边是来自于outlier
            if(pFrame->mvbOutlier[idx])
            {
                e->ComputeResidual(); 
            }

            // 就是error*Omega*error,表征了这个点的误差大小(考虑置信度以后)
            const double chi2 = e->Chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);                 // 设置为outlier , level 1 对应为外点,上面的过程中我们设置其为不优化
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);                 // 设置为inlier, level 0 对应为内点,上面的过程中我们就是要优化这些关系
            }

            if(it==2)
                e->SetLossFunction(nullptr);   // 除了前两次优化需要RobustKernel以外, 其余的优化都不需要 -- 因为重投影的误差已经有明显的下降了
        }

        if(problem.getEdgeSize()<10)
            break;

    } // 一共要进行四次优化

    // Recover optimized pose and return number of inliers
    // Step 5 得到优化后的当前帧的位姿
    // tx, ty, tz, qx, qy, qz, qw

    Vec7 tq = vSE3->Parameters();
    // q的初始化顺序wxyz，实际存储顺序xyzw
    cv::Mat update_pose = Converter::toCvMat( Qd(tq[6],tq[3],tq[4],tq[5]), Vec3(tq.head<3>())); 
    pFrame->SetPose(update_pose);

    std::cout << "             ::PoseOptimization() 误差边数量:" << problem.getEdgeSize()
              << "   此次优化位姿为: " << update_pose.row(0) << std::endl
              << "                                                                   " << update_pose.row(1) << std::endl
              << "                                                                   " << update_pose.row(2) << std::endl;

    nBad=0;
    // 优化结束,开始遍历参与优化的每一条误差边
    for(size_t i=0, iend=vpEdges.size(); i<iend; i++)
    {
        shared_ptr<backend::EdgeReprojectionPoseOnly> e = vpEdges[i];

        const size_t idx = vnIndexEdge[i];

        // 如果这条误差边是来自于outlier
        if(pFrame->mvbOutlier[idx])
        {
            e->ComputeResidual();
        }

        // 就是error*Omega*error,表征了这个点的误差大小(考虑置信度以后)
        const double chi2 = e->Chi2();

        if(chi2>chi2Mono[0])
        {
            pFrame->mvbOutlier[idx]=true;
            nBad++;
        }
        else
        {
            pFrame->mvbOutlier[idx]=false;
        }

    }

    // 并且返回内点数目
    return nInitialCorrespondences-nBad;
}
}