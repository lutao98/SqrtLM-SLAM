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
    shared_ptr<backend::VertexPose> vPose(new backend::VertexPose());
    // parameters: tx, ty, tz, qx, qy, qz, qw, 7 DoF
    vPose->SetParameters(Converter::toEigenVecTQ(pFrame->mTcw));
    vPose->SetFixed(false);
    problem.AddVertex(vPose);

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
            e->AddVertex(vPose);
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
        // parameters: tx, ty, tz, qx, qy, qz, qw, 7 DoF
        vPose->SetParameters(Converter::toEigenVecTQ(pFrame->mTcw));    
     
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

    Vec7 tq = vPose->Parameters();
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

/**
 * @brief Local Bundle Adjustment
 *
 * 1. Vertex:
 *     - backend::VertexPose()，LocalKeyFrames，即当前关键帧的位姿、与当前关键帧相连的关键帧的位姿
 *     - backend::VertexPose()，FixedCameras，即能观测到LocalMapPoints的关键帧（并且不属于LocalKeyFrames）的位姿，在优化中这些关键帧的位姿不变
 *     - backend::VertexPointXYZ()，LocalMapPoints，即LocalKeyFrames能观测到的所有MapPoints的位置
 * 2. Edge:
 *     - backend::EdgeReprojectionXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param pKF        KeyFrame
 * @param pbStopFlag 是否停止优化的标志
 * @param pMap       在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 * @note 由局部建图线程调用,对局部地图进行优化的函数
 */
void MyOptimizer::LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap){
    
    // 该优化函数用于LocalMapping线程的局部BA优化
    // Local KeyFrames: First Breadth Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    // Step 1 将当前关键帧及其共视关键帧加入lLocalKeyFrames
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    // 找到关键帧连接的共视关键帧（一级相连），加入lLocalKeyFrames中
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];

        // 把参与局部BA的每一个关键帧的 mnBALocalForKF设置为当前关键帧的mnId，防止重复添加
        pKFi->mnBALocalForKF = pKF->mnId;

        // 保证该关键帧有效才能加入
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    // Step 2 遍历 lLocalKeyFrames 中关键帧，将它们观测的MapPoints加入到lLocalMapPoints
    list<MapPoint*> lLocalMapPoints;
    // 遍历 lLocalKeyFrames 中的每一个关键帧
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        // 取出该关键帧对应的地图点
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        // 遍历这个关键帧观测到的每一个地图点，加入到lLocalMapPoints
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
            {
                if(!pMP->isBad())   //保证地图点有效
                {
                    // 把参与局部BA的每一个地图点的 mnBALocalForKF设置为当前关键帧的mnId
                    // mnBALocalForKF 是为了防止重复添加
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
                }
            }   // 判断这个地图点是否靠谱
        } // 遍历这个关键帧观测到的每一个地图点
    } // 遍历 lLocalKeyFrames 中的每一个关键帧


    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // Step 3 得到能被局部MapPoints观测到，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时不优化
    list<KeyFrame*> lFixedCameras;
    // 遍历局部地图中的每个地图点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        // 观测到该MapPoint的KF和该MapPoint在KF中的索引
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        // 遍历所有观测到该地图点的关键帧
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            // pKFi->mnBALocalForKF!=pKF->mnId 表示不属于局部关键帧，
            // pKFi->mnBAFixedForKF!=pKF->mnId 表示还未标记为fixed（固定的）关键帧
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                // 将局部地图点能观测到的、但是不属于局部BA范围的关键帧的mnBAFixedForKF标记为pKF（触发局部BA的当前关键帧）的mnId
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }


    // Setup optimizer
    // step 4 构建 problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    problem.setStorageMode(backend::Problem::StorageMode::GENERIC_MODE);
    problem.setDrawHessian(false);
    problem.setDebugOutput(true);       //调试输出:迭代 耗时等信息

    // 外界设置的停止优化标志
    // 可能在 Tracking::NeedNewKeyFrame() 里置位,指针的值随时变
    if(pbStopFlag)
        problem.setForceStopFlag(pbStopFlag);

    // Keyframe->id 到 VertexPose 的hash
    std::unordered_map<unsigned long, shared_ptr<backend::VertexPose>> id2vPose;
    // MapPoint->id 到 VertexPointXYZ 的hash
    std::unordered_map<unsigned long, shared_ptr<backend::VertexPointXYZ>> id2vPoint;

    // Set Local KeyFrame vertices
    // Step 5 添加待优化的位姿顶点：Pose of Local KeyFrame
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame *pKFi = *lit;

        shared_ptr<backend::VertexPose> vPose(new backend::VertexPose());
        // 设置初始优化位姿 parameters: tx, ty, tz, qx, qy, qz, qw, 7 DoF
        vPose->SetParameters(Converter::toEigenVecTQ(pKFi->GetPose()));
        // 如果是初始关键帧，要fix住位姿不优化
        vPose->SetFixed(pKFi->mnId==0);
        problem.AddVertex(vPose);

        id2vPose[pKFi->mnId] = vPose;
    }

    // Set Fixed KeyFrame vertices
    // Step  6 添加不优化的位姿顶点：Pose of Fixed KeyFrame，注意这里调用了vFixPose->setFixed(true)
    // 不优化为啥也要添加？回答：为了增加约束信息
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        shared_ptr<backend::VertexPose> vFixPose(new backend::VertexPose());
        vFixPose->SetParameters(Converter::toEigenVecTQ(pKFi->GetPose()));

        // 如果是初始关键帧，要fix住位姿不优化
        vFixPose->SetFixed(true);
        problem.AddVertex(vFixPose);

        id2vPose[pKFi->mnId] = vFixPose;
    }

    // Set MapPoint vertices
    // Step  7 添加待优化的3D地图点顶点
    // 边的数目 = pose数目 * 地图点数目
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<shared_ptr<backend::EdgeReprojectionXYZ>> vpEdges;
    vpEdges.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKF;
    vpEdgeKF.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdge;
    vpMapPointEdge.reserve(nExpectedSize);

    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
    const float thHuberMono = sqrt(5.991);
    // 自由度为3的卡方分布，显著性水平为0.05，对应的临界阈值7.815
    const float thHuberStereo = sqrt(7.815);

    // 核函数
    backend::LossFunction *lossfunction;
    lossfunction = new backend::CauchyLoss(thHuberMono);
    //    lossfunction = new backend::TukeyLoss(1.0);

    // 遍历所有的局部地图中的地图点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        // 添加顶点：MapPoint
        MapPoint* pMP = *lit;
        shared_ptr<backend::VertexPointXYZ> vPoint(new backend::VertexPointXYZ());
        vPoint->SetParameters(Converter::toVector3d(pMP->GetWorldPos()));
        problem.AddVertex(vPoint);

        id2vPoint[pMP->mnId] = vPoint;

        // 观测到该地图点的KF和该地图点在KF中的索引
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        // Set edges
        // Step 8 在添加完了一个地图点之后, 对每一对关联的MapPoint和KeyFrame构建边
        // 遍历所有观测到当前地图点的关键帧
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                shared_ptr<backend::EdgeReprojectionXYZ> e(new backend::EdgeReprojectionXYZ(obs));
                // 注意：verticies_顶点顺序必须为 XYZ、Tcw ==> verticies_[0]、verticies_[1]
                e->AddVertex(vPoint);
                e->AddVertex(id2vPose[pKFi->mnId]);

                // 权重为特征点所在图像金字塔的层数的倒数
                const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                e->SetInformation(Eigen::Matrix2d::Identity()*invSigma2);
                // 在这里使用了鲁棒核函数
                e->SetLossFunction(lossfunction);
                // 设置相机内参
                e->setCamIntrinsics(pKFi->fx,pKFi->fy,pKFi->cx,pKFi->cy);

                problem.AddEdge(e);

                vpEdges.push_back(e);
                vpEdgeKF.push_back(pKFi);
                vpMapPointEdge.push_back(pMP);
            } 
        } // 遍历所有观测到当前地图点的关键帧
    } // 遍历所有的局部地图中的地图点


    // 开始BA前再次确认是否有外部请求停止优化，因为这个变量是引用传递，会随外部变化
    // 可能在 Tracking::NeedNewKeyFrame(), mpLocalMapper->InsertKeyFrame 里置位
    if(pbStopFlag)
        if(*pbStopFlag){
            std::cout << "                 ::开始BA(1)，确认有外部请求停止优化！退出" << std::endl;
            return;
        }
    
    // Step 9 开始优化,分成两个阶段
    // 第一阶段优化
    problem.setOptimizeLevel(0);
    // 迭代5次
    problem.Solve(5);

    bool bDoMore= true;

    // 检查是否外部请求停止
    if(pbStopFlag)
        if(*pbStopFlag){
            bDoMore = false;
            std::cout << "                 ::开始BA(2)，确认有外部请求停止优化！退出" << std::endl;
        }
    
    // 如果有外部请求停止,那么就不在进行第二阶段的优化
    if(bDoMore)
    {
        // Check inlier observations
        // Step 10 检测outlier，并设置下次不优化
        for(size_t i=0, iend=vpEdges.size(); i<iend;i++)
        {
            shared_ptr<backend::EdgeReprojectionXYZ> e = vpEdges[i];
            MapPoint* pMP = vpMapPointEdge[i];

            if(pMP->isBad())
                continue;

            // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
            // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
            // 如果 当前边误差超出阈值，或者边链接的地图点深度值为负，说明这个边有问题，不优化了。
            if(e->Chi2()>5.991 || !e->isDepthPositive())
            {
                // 不优化
                e->setLevel(1);
            }
            // 第二阶段优化的时候就属于精求解了,所以就不使用核函数
            // e->SetLossFunction(nullptr);         // 记得改回去
        }

        // Optimize again without the outliers
        // Step 11：排除误差较大的outlier后再次优化 -- 第二阶段优化
        problem.setOptimizeLevel(0);
        problem.Solve(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdges.size());

    // Check inlier observations
    // Step 12：在优化后重新计算误差，剔除连接误差比较大的关键帧和MapPoint
    // 对于单目误差边
    for(size_t i=0, iend=vpEdges.size(); i<iend;i++)
    {
        shared_ptr<backend::EdgeReprojectionXYZ> e = vpEdges[i];
        MapPoint* pMP = vpMapPointEdge[i];

        if(pMP->isBad())
            continue;

        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
        // 如果 当前边误差超出阈值，或者边链接的地图点深度值为负，说明这个边有问题，要删掉了
        if(e->Chi2()>5.991 || !e->isDepthPositive())
        {
            // outlier
            KeyFrame* pKFi = vpEdgeKF[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // 删除点
    // 连接偏差比较大，在关键帧中剔除对该地图点的观测
    // 连接偏差比较大，在地图点中剔除对该关键帧的观测
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;

            pKFi->EraseMapPointMatch(pMPi);

            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    // Step 13：优化后更新关键帧位姿以及地图点的位置、平均观测方向等属性
    // Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        shared_ptr<backend::VertexPose> vPose = id2vPose[pKFi->mnId];
        Vec7 tq = vPose->Parameters();
        // q的初始化顺序wxyz，实际存储顺序xyzw
        cv::Mat update_pose = Converter::toCvMat( Qd(tq[6],tq[3],tq[4],tq[5]), Vec3(tq.head<3>())); 

        pKFi->SetPose(update_pose);
        if(lit==lLocalKeyFrames.begin()){
            std::cout << "                 ::myLocalBundleAdjustment() 误差边数量:" << problem.getEdgeSize() << std::endl
                      << "                 ::局部BA结束，最新关键帧ID:" << pKFi->mnId << std::endl
                      << "                 ::最新关键帧位姿:" << pKFi->GetPose().row(0)  << std::endl
                      << "                                  " << pKFi->GetPose().row(1)  << std::endl
                      << "                                  " << pKFi->GetPose().row(2)  << std::endl;
        }
    }

    // Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        shared_ptr<backend::VertexPointXYZ> vPoint = id2vPoint[pMP->mnId];
        Vec3 updatePoint(vPoint->Parameters());
        pMP->SetWorldPos(Converter::toCvMat(updatePoint));
        pMP->UpdateNormalAndDepth();
    }
}

}