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

//主进程的实现文件

#include "System.h"
#include "utils/settings.h"
#include <thread>            //多线程
#include <iomanip>           //主要是对cin,cout之类的一些操纵运算子
#include <unistd.h>

namespace ORB_SLAM2
{
Settings settings;

SLAMresult System::getSLAMresult(){
  SLAMresult result(mpMap->GetAllKeyFrames(),
                    GetTrackedKeyPointsUn(),
                    mpTracker->KP_Viewer_,
                    mpTracker->mCurrentFrame.mvDepth,
                    mpTracker->mlpReferences,
                    mpTracker->mlRelativeFramePoses,
                    mpMap->GetAllMapPoints(),
                    mpMap->GetReferenceMapPoints(),
                    mpTracker->getTrackingLocalKFid(),
                    mpTracker->getDepthimg(),
                    mpTracker->getRangeImage(),
                    mpTracker->getCornerPoints(),
                    mpTracker->getFlatPoints(),
                    mpTracker->getLessCornerPoints(),
                    mpTracker->getLessFlatPoints(),
                    mpTracker->getLidarLocalMap());
  sort(result.vpKFs_.begin(), result.vpKFs_.end(), ORB_SLAM2::KeyFrame::lId);
  return result;
}

//系统的构造函数，将会启动其他的线程
System::System(const string &strVocFile,               //词典文件路径
               const string &strSettingsFile,          //配置文件路径
               const eSensor sensor,                   //传感器类型
               const bool bUseViewer):                 //是否使用可视化界面
    mSensor(sensor),                                   //初始化传感器类型
    mbReset(false),                                    //无复位标志
    mbActivateLocalizationMode(false),                 //没有这个模式转换标志
    mbDeactivateLocalizationMode(false)                //没有这个模式转换标志
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl <<
    "LT-SLAM  Copyright (C) 2020 Tao Lu, University of HuNan" << endl << endl;

    // 输出当前传感器类型
    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;
    else if(mSensor==FUSION)
        cout << "FUSION" << endl;

    lidar_config_ = NULL;

    // Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), 	// 将配置文件名转换成为字符串
                               cv::FileStorage::READ);		// 只读
    // 如果打开失败，就输出调试信息
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       //然后退出
       exit(-1);
    }

    settings.fx = fsSettings["Camera.fx"];
    settings.fy = fsSettings["Camera.fy"];
    settings.cx = fsSettings["Camera.cx"];
    settings.cy = fsSettings["Camera.cy"];

    // Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    // 建立一个新的ORB字典
    mpVocabulary = new ORBVocabulary();
    // 获取字典加载状态
    //bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    bool bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);   //bin加载
    // 如果加载失败，就输出调试信息
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        // 然后退出
        exit(-1);
    }
    // 否则则说明加载成功
    cout << "Vocabulary loaded!" << endl << endl;

    // Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    // Create the Map
    mpMap = new Map();

    // 在本主进程中初始化追踪线程
    // Initialize the Tracking thread
    // (it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this,                   //指向这个system
                             mpVocabulary,           //字典
                             mpMap,                  //地图
                             mpKeyFrameDatabase,     //关键帧地图
                             strSettingsFile,        //设置文件路径
                             mSensor);               //传感器类型iomanip

    // 初始化局部建图线程并运行
    // Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap,                  //指定使iomanip
                                     mSensor==MONOCULAR,
                                     &lidar_config_);

    // 运行这个局部建图线程
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,    //这个线程会调用的函数
                                 mpLocalMapper);                   //这个调用函数的参数

    // Initialize the Loop Closing thread and launchiomanip
    mpLoopCloser = new LoopClosing(mpMap,                        //地图
                                   mpKeyFrameDatabase,           //关键帧数据库
                                   mpVocabulary,                 //ORB字典
                                   mSensor!=MONOCULAR);          //当前的传感器是否是单目
    // 创建回环检测线程
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run,   //线程的主函数
                                mpLoopCloser);                  //该函数的参数

    // Set pointers between threads
    // 设置进程间的指针
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}


void System::setCalibrationParam(Eigen::Matrix<double,3,4> intrinsicMatrix,
                                 Eigen::Matrix<double,4,4> extrinsicMatrix){
  intrinsicMatrix_=intrinsicMatrix;
  extrinsicMatrix_=extrinsicMatrix;
}
Eigen::Matrix<double,3,4> System::getIntrinsicMatrix(){
  return intrinsicMatrix_;
}
Eigen::Matrix<double,4,4> System::getExtrinsicMatrix(){
  return extrinsicMatrix_;
}


// 同理，输入为单目图像时的追踪器接口
cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        // 独占锁，主要是为了mbActivateLocalizationMode和mbDeactivateLocalizationMode不会发生混乱
        unique_lock<mutex> lock(mMutexMode);
        // mbActivateLocalizationMode为true会关闭局部地图线程
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            // 局部地图关闭以后，只进行追踪的线程，只计算相机的位姿，没有对局部地图进行更新
            // 设置mbOnlyTracking为真
            mpTracker->InformOnlyTracking(true);
            // 关闭线程可以使得别的线程得到更多的资源
            mbActivateLocalizationMode = false;
        }
        // 如果mbDeactivateLocalizationMode是true，局部地图线程就被释放, 关键帧从局部地图中删除.
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
        }
    }

    // 获取相机位姿的估计结果
    cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}


cv::Mat System::TrackFusion(const cv::Mat &im,
                            pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_inputPtr,
                            const double &timestamp){
//    if(mSensor!=FUSION){
//        cerr << "ERROR: you called Trackfusion but input sensor was not set to fusion(camera and lidar)." << endl;
//        exit(-1);
//    }

    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode){
            mpLocalMapper->RequestStop();
            while(!mpLocalMapper->isStopped()){
                usleep(1000);
            }
            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode){
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset){
            mpTracker->Reset();
            mbReset = false;
        }
    }

    // Grab Image Fusion
    // 获得相机位姿的估计
    cv::Mat Tcw = mpTracker->GrabImageFusion(im, lidar_inputPtr, timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeys;   // is mvkeyun

    vector<MapPoint*> local_mappoints=mpMap->GetReferenceMapPoints();
    int valid_MPcount=0;
    for(MapPoint* pMP:local_mappoints){
        if(pMP){
            if(!pMP->isBad())
                valid_MPcount++;
        }
    }
    std::cout << "[Result]::地图点数量: " << mpMap->MapPointsInMap()
              << "    参考地图点数量:" << local_mappoints.size()
              << "    有效参考地图点数量:" << valid_MPcount << std::endl;
    std::cout << "[Result]::位姿:" << Tcw.row(0) << std::endl
              << "               " << Tcw.row(1) << std::endl
              << "               " << Tcw.row(2) << std::endl;

    return Tcw;
}


// 激活定位模式
void System::ActivateLocalizationMode()
{
    // 上锁
    unique_lock<mutex> lock(mMutexMode);
    // 设置标志
    mbActivateLocalizationMode = true;
}


// 取消定位模式
void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

// 判断是否地图有较大的改变
bool System::MapChanged()
{
    static int n=0;
    // 其实整个函数功能实现的重点还是在这个GetLastBigChangeIdx函数上
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}


// 准备执行复位
void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}


// 退出
void System::Shutdown()
{
    // 对局部建图线程和回环检测线程发送终止请求
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || 
          !mpLoopCloser->isFinished()  ||
          mpLoopCloser->isRunningGBA())        //TODO isRunningGBA函数是用来做什么的？
    {
        usleep(5000);
    }

}


//按照TUM格式保存相机运行轨迹并保存到指定的文件中
void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    // 只有在传感器为双目或者RGBD时才可以工作
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    // 从地图中获取所有的关键帧
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    // 根据关键帧生成的先后顺序（id）进行排序
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    // 到原点的转换，获取这个转换矩阵
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    // 文件写入的准备工作
    ofstream f;
    f.open(filename.c_str());
    // 这个可以理解为，在输出浮点数的时候使用0.3141592654这样的方式而不是使用科学计数法
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.
    // 之前的帧位姿都是基于其参考关键帧的，现在我们把它恢复

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    // 参考关键帧列表
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    // 所有帧对应的时间戳列表
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    // 每帧的追踪状态组成的列表
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    // 对于每一个mlRelativeFramePoses中的帧lit
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();
        lit!=lend;
        lit++, lRit++, lT++, lbL++)		// TODO 为什么是在这里更新参考关键帧？
    {
        // 如果该帧追踪失败，不管它，进行下一个
        if(*lbL)
            continue;

        // 获取其对应的参考关键帧
        KeyFrame* pKF = *lRit;

        // 变换矩阵的初始化，初始化为一个单位阵
        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled（剔除）, traverse（扫描？） the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            // 更新关键帧变换矩阵的初始值，
            Trw = Trw*pKF->mTcp;
            // 并且更新到原关键帧的父关键帧
            pKF = pKF->GetParent();
        }// 查看当前使用的参考关键帧是否为bad
        // TODO 其实我也是挺好奇，为什么在这里就能够更改掉不合适的参考关键帧了呢

        // TODO 这里的函数GetPose()和上面的mTcp有什么不同？
        // 最后一个Two是原点校正
        // 最终得到的是参考关键帧相对于世界坐标系的变换
        Trw = Trw*pKF->GetPose()*Two;

        // 在此基础上得到相机当前帧相对于世界坐标系的变换
        cv::Mat Tcw = (*lit)*Trw;
        // 然后分解出旋转矩阵
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        // 以及平移向量
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        // 用四元数表示旋转
        vector<float> q = Converter::toQuaternion(Rwc);

        // 然后按照给定的格式输出到文件中
        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }////对于每一个mlRelativeFramePoses中的帧lit所进行的操作

    // 操作完毕，关闭文件并且输出调试信息
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


//保存关键帧的轨迹
void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    // 获取关键帧vector并按照生成时间对其进行排序
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // 本来这里需要进行原点校正，但是实际上没有做
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    // cv::Mat Two = vpKFs[0]->GetPoseInverse();

    // 文件写入的准备操作
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // 对于每个关键帧
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        // 获取该 关键帧
        KeyFrame* pKF = vpKFs[i];

        // 原本有个原点校正，这里注释掉了
        // pKF->SetPose(pKF->GetPose()*Two);

        // 如果这个关键帧是bad那么就跳过
        if(pKF->isBad())
            continue;

        // 抽取旋转部分和平移部分，前者使用四元数表示
        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        // 按照给定的格式输出到文件中
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    // 关闭文件
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


// 按照KITTI数据集的格式将相机的运动轨迹保存到文件中
void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "轨迹保存至:" << filename << " ..." << endl;
    // 检查输入数据的类型
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    // 下面的操作和前面TUM数据集格式的非常相似，因此不再添加注释
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    cout << endl << "轨迹长度:" << mpTracker->mlpReferences.size() << endl;
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end(); lit!=lend; lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
            //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "轨迹已保存！" << endl;
}


//获取追踪器状态
int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}


//获取追踪到的地图点（其实实际上得到的是一个指针）
vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}


//获取追踪到的关键帧的点
vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}


//保存关键帧的轨迹
vector<cv::Mat> System::GetTrajectory()
{
    vector<cv::Mat> trajectory;

    // 获取关键帧vector并按照生成时间对其进行排序
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // 本来这里需要进行原点校正，但是实际上没有做
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    // cv::Mat Two = vpKFs[0]->GetPoseInverse();

    // 对于每个关键帧
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        // 获取该 关键帧
        KeyFrame* pKF = vpKFs[i];

        // 原本有个原点校正，这里注释掉了
        // pKF->SetPose(pKF->GetPose()*Two);

        // 如果这个关键帧是bad那么就跳过
        if(pKF->isBad())
            continue;

        cv::Mat t = pKF->GetCameraCenter();

        trajectory.push_back(t);

    }
    return trajectory;
}

} //namespace ORB_SLAM
