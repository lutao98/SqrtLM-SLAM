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

/**
 * @file Frame.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 帧的实现文件
 * @version 0.1
 * @date 2019-01-03
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "data_structure/Frame.h"
#include "utils/Converter.h"
#include "utils/tic_toc.h"
#include <thread>
#include <pcl/common/transforms.h>

namespace ORB_SLAM2
{

//下一个生成的帧的ID,这里是初始化类的静态成员变量
long unsigned int Frame::nNextId=0;

//是否要进行初始化操作的标志
//这里给这个标志置位的操作是在最初系统开始加载到内存的时候进行的，下一帧就是整个系统的第一帧，所以这个标志要置位
bool Frame::mbInitialComputations=true;

//TODO 下面这些都没有进行赋值操作，但是也写在这里，是为什么？可能仅仅是前视声明?
//目测好像仅仅是对这些类的静态成员变量做个前视声明，没有发现这个操作有特殊的含义
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//无参的构造函数默认为空
Frame::Frame(){}

/** @details 另外注意，调用这个函数的时候，这个函数中隐藏的this指针其实是指向目标帧的
 */
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), 
     mpORBextractorLeft(frame.mpORBextractorLeft), 
     mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), 
     mK(frame.mK.clone()),									//深拷贝
     mDistCoef(frame.mDistCoef.clone()),					//深拷贝
     mbf(frame.mbf), 
     mb(frame.mb), 
     mThDepth(frame.mThDepth), 
     N(frame.N), 
     mvKeys(frame.mvKeys),									//经过实验，确定这种通过同类型对象初始化的操作是具有深拷贝的效果的
     mvKeysRight(frame.mvKeysRight), 						//深拷贝
     mvKeysUn(frame.mvKeysUn),  							//深拷贝
     mvuRight(frame.mvuRight),								//深拷贝
     mvDepth(frame.mvDepth), 								//深拷贝
     mBowVec(frame.mBowVec), 								//深拷贝
     mFeatVec(frame.mFeatVec),								//深拷贝
     mDescriptors(frame.mDescriptors.clone()), 				//cv::Mat深拷贝
     mDescriptorsRight(frame.mDescriptorsRight.clone()),	//cv::Mat深拷贝
     mvpMapPoints(frame.mvpMapPoints), 						//深拷贝
     mvbOutlier(frame.mvbOutlier), 							//深拷贝
     mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), 
     mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), 
     mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), 					//深拷贝
     mvInvScaleFactors(frame.mvInvScaleFactors),			//深拷贝
     mvLevelSigma2(frame.mvLevelSigma2), 					//深拷贝
     mvInvLevelSigma2(frame.mvInvLevelSigma2),				//深拷贝

     Depthimg_(frame.Depthimg_.clone()),
     range_img_visual_(frame.range_img_visual_.clone()),
     lidar_inputPtr_(frame.lidar_inputPtr_),
     all_laser_scans_(frame.all_laser_scans_),
     laser_scans_(frame.laser_scans_),
     // 每行范围
     scan_ranges_(frame.scan_ranges_),
     range_image_(frame.range_image_),
     image_index_(frame.image_index_),
     surface_points_less_flat_index_(frame.surface_points_less_flat_index_),
     ringAng(frame.ringAng),
     scanAng(frame.scanAng),
     cloud_in_rings_(frame.cloud_in_rings_),
     corner_points_sharp_(frame.corner_points_sharp_),
     corner_points_less_sharp_(frame.corner_points_less_sharp_),
     surface_points_flat_(frame.surface_points_flat_),
     surface_points_less_flat_(frame.surface_points_less_flat_),
     surface_points_flat_normal_(frame.surface_points_flat_normal_),
     surface_points_less_flat_normal_(frame.surface_points_less_flat_normal_),
     curvature_idx_pairs_(frame.curvature_idx_pairs_),
     less_sharp_(frame.less_sharp_),
     sharp_(frame.sharp_),
     scan_ring_mask_(frame.scan_ring_mask_),
     lidarconfig_(frame.lidarconfig_)
{
    //逐个复制，其实这里也是深拷贝
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            //这里没有使用前面的深拷贝方式的原因可能是mGrid是由若干vector类型对象组成的vector，
            //但是自己不知道vector内部的源码不清楚其赋值方式，在第一维度上直接使用上面的方法可能会导致
            //错误使用不合适的复制函数，导致第一维度的vector不能够被正确地“拷贝”
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        //这里说的是给新的帧设置Pose
        SetPose(frame.mTcw);
}


/**
 * @brief 单目帧构造函数
 * 
 * @param[in] imGray                            //灰度图
 * @param[in] timeStamp                         //时间戳
 * @param[in & out] extractor                   //ORB特征点提取器的句柄
 * @param[in] voc                               //ORB字典的句柄
 * @param[in] K                                 //相机的内参数矩阵
 * @param[in] distCoef                          //相机的去畸变参数
 * @param[in] bf                                //baseline*f
 * @param[in] thDepth                           //区分远近点的深度阈值
 */
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor,
             ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    // Step 1 帧的ID 自增
    mnId=nNextId++;

    lidarconfig_=NULL;

    // Step 2 计算图像金字塔的参数 
    // Scale Level Info
    //获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    //获取每层的缩放因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    //计算每层缩放因子的自然对数
    mfLogScaleFactor = log(mfScaleFactor);
    //获取各层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    //获取各层图像的缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    //获取sigma^2
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    //获取sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // Step 3 对这个单目图像进行提取特征点, 第一个参数0-左图， 1-右图
    ExtractORB(0,imGray);

    //求出特征点的个数
    N = mvKeys.size();

    //如果没有能够成功提取出特征点，那么就直接返回了
    if(mvKeys.empty())
        return;

    // Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正 
    UndistortKeyPoints();

    // Set no stereo information
    // 由于单目相机无法直接获得立体信息，所以这里要给右图像对应点和深度赋值-1表示没有相关信息
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    // 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    // 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    //  Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
        // 计算去畸变后图像的边界
        ComputeImageBounds(imGray);

        // 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        // 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        //给类的静态成员变量复制
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        // 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        //特殊的初始化过程完成，标志复位
        mbInitialComputations=false;
    }

    //计算 basline
    mb = mbf/fx;

    // 将特征点分配到图像网格中
    AssignFeaturesToGrid();
}


/**
 * @brief 为FUSION准备的帧构造函数
 *
 * @param[in] imGray        对RGB图像灰度化之后得到的灰度图像
 * @param[in] imDepth       LIDAR深度图像
 * @param[in] timeStamp     时间戳
 * @param[in] extractor     特征点提取器句柄
 * @param[in] voc           ORB特征点词典的句柄
 * @param[in] K             相机的内参数矩阵
 * @param[in] distCoef      相机的去畸变参数
 * @param[in] bf            baseline*bf
 * @param[in] thDepth       远点和近点的深度区分阈值
 */
Frame::Frame(const cv::Mat &imGray, pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_inputPtr,
             const lidarConfig* lidarconfig, const double &timeStamp,
             ORBextractor* extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef,
             const float &bf, const float &thDepth,
             const Eigen::Matrix<double,3,4> &intrinsicMatrix,
             const Eigen::Matrix<double,4,4> &extrinsicMatrix)
    :mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Step 1 帧的ID 自增
    mnId=nNextId++;
    lidar_inputPtr_=lidar_inputPtr;
    lidarconfig_=lidarconfig;

    range_img_visual_ = cv::Mat(lidarconfig_->row_num_, lidarconfig_->col_num_, CV_64F, cv::Scalar::all(0));
    laser_scans_.clear();
    for (int i = 0; i < lidarconfig_->row_num_; ++i) {
      PointIRTCloudPtr scan(new PointIRTCloud());
      laser_scans_.push_back(scan);
    }
    range_image_.clear();
    for (int i = 0; i < lidarconfig_->row_num_; ++i) {
      std::vector<ImageElement> image_row(lidarconfig_->col_num_);
      range_image_.push_back(image_row);
    }
    image_index_.resize(lidarconfig_->row_num_);

    // Step 2 计算图像金字塔的参数
    // Scale Level Info
    // 获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    // 获取每层的缩放因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    // 计算每层缩放因子的自然对数
    mfLogScaleFactor = log(mfScaleFactor);
    // 获取各层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    // 获取各层图像的缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    // 获取sigma^2
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    // 获取sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    TicToc sum_timer;
    std::thread loopthread(&Frame::lidarProcess, this);

    TicToc visual_timer;
    /// 处理图像
    {
    Depthimg_ = cv::Mat(imGray.rows, imGray.cols, CV_64F, cv::Scalar::all(0));
    //将lidar先验通过外参转换到深度图
    int depthpixel_num=0;
    for(int i = 0; i < lidar_inputPtr->size(); i++){

        if(lidar_inputPtr->points[i].x<1)  continue;

        Eigen::Vector4d P_lidar(lidar_inputPtr->points[i].x,
                                lidar_inputPtr->points[i].y,
                                lidar_inputPtr->points[i].z,
                                1);
        Eigen::Vector3d z_P_uv = intrinsicMatrix*extrinsicMatrix*P_lidar;
        Eigen::Vector3i P_uv = Eigen::Vector3i( int( z_P_uv[0]/z_P_uv[2] ), int( z_P_uv[1]/z_P_uv[2] ), 1 );

        if(P_uv[0] >= 0 && P_uv[1] >= 0 && P_uv[0]<=imGray.cols-1 && P_uv[1]<=imGray.rows-1){
            cv::Point2i pixel;
            pixel.x = P_uv[0];
            pixel.y = P_uv[1];
            if(lidar_inputPtr->points[i].x>1){
                Depthimg_.at<double>(pixel.y,pixel.x)=(extrinsicMatrix*P_lidar).z();
                depthpixel_num++;
            }
        }
    }
    /** 3. 提取彩色图像(其实现在已经灰度化成为灰度图像了)的特征点 \n Frame::ExtractORB() */

    // ORB extraction
    // Step 3 对图像进行提取特征点, 第一个参数0-左图，1-右图
    ExtractORB(0,imGray);

    // 获取特征点的个数
    N = mvKeys.size();

    // 如果这一帧没有能够提取出特征点，那么就直接返回了
    if(mvKeys.empty())
        return;

    // Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正
    UndistortKeyPoints();

    // !!!!
    // Set no stereo information
    // 由于单目相机无法直接获得立体信息，所以这里要给右图像对应点和深度赋值-1表示没有相关信息
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    /// 改
    // 根据雷达深度补充特帧点深度，这里会补充深度较为稳定的特帧点
    {
        const int half_patch_width=4;
        const int half_patch_height=7;
        int min_v=0;
        for(int v=0; v<Depthimg_.rows; v++){
            bool break_flag=false;
            for(int u=0; u<Depthimg_.cols; u++){
                double pixelvalue=Depthimg_.at<double>(v, u);
                if(pixelvalue>0){
                    min_v=v;
                    break_flag=true;
                    break;
                }
            }
            if(break_flag) break;
        }

        for(int i=0; i<mvKeys.size(); i++){
            if(mvKeys[i].pt.y<min_v){
                mvKeys[i].class_id=0;          // 标记0代表特帧点邻域内没有点云
            }else{
                double min_pixel_distance=100000;
                cv::Point2i nearest_pixel;
                bool exist_nearest=false, large_depth_range=false;
                std::vector<double> depthrange;
                for(int dx=-half_patch_width; dx<half_patch_width; dx++){
                    for(int dy=-half_patch_height; dy<half_patch_height; dy++){

                      uchar pixelvalue=Depthimg_.at<double>(mvKeys[i].pt.y+dy, mvKeys[i].pt.x+dx);
                      if(pixelvalue==0)    continue;

                      double depth=Depthimg_.at<double>(mvKeys[i].pt.y+dy, mvKeys[i].pt.x+dx);
                      depthrange.push_back(depth);

                      double pixel_distance=sqrt(dy*dy+dx*dx);
                      if(pixel_distance<min_pixel_distance){
                          min_pixel_distance=pixel_distance;
                          exist_nearest=true;
                          nearest_pixel.x=mvKeys[i].pt.x+dx;
                          nearest_pixel.y=mvKeys[i].pt.y+dy;
                      }
                    }
                }

                if(exist_nearest){

                    sort(depthrange.begin(),depthrange.end());
                    //小框内如果像素点深度差大于阈值就认为是深度估计不稳定的特帧点
                    if(depthrange.back()-depthrange.front()>2){
                        large_depth_range=true;
                    }

                    if(!large_depth_range){
                        mvKeys[i].class_id=1;       // 标记1代表特帧点深度比较稳定
                        mvDepth.at(i)=Depthimg_.at<double>(nearest_pixel.y, nearest_pixel.x);
                    }else{
                        mvKeys[i].class_id=2;       // 标记2代表特帧点深度不稳定
//                        mvDepth.at(i)=lidarDepth.at<double>(nearest_pixel.y, nearest_pixel.x);
                    }

                }else{
                    mvKeys[i].class_id=0;           // 标记3代表特帧点邻域内没有点云
                }
            }
        }

        int existdepthPnum=0;
        for(float depth:mvDepth){
            if(depth>0)
                existdepthPnum++;
        }
        std::cout << "[Frame]::特帧点数量:" << mvDepth.size()
                  << "   存在深度的像素数量:" << depthpixel_num
                  << "   存在深度的特帧点数量:" << existdepthPnum << std::endl;
        std::cout << "       ::激光角点数量: " << corner_points_sharp_.size()
                  << "   激光弱角点数量: " << corner_points_less_sharp_.size() << std::endl
                  << "       ::激光平面点数量: " << surface_points_flat_.size()
                  << "   激光弱平面点数量: " << surface_points_less_flat_.size() << std::endl;
    }
    }

    // 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    // 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    // Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
        // 计算去畸变后图像的边界
        ComputeImageBounds(imGray);

        // 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        // 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        // 给类的静态成员变量复制
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        // 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        // 特殊的初始化过程完成，标志复位
        mbInitialComputations=false;
    }

    // 计算假想的基线长度 baseline= mbf/fx
    // 后面要对从RGBD相机输入的特征点,结合相机基线长度,焦距,以及点的深度等信息来计算其在假想的"右侧图像"上的匹配点
    mb = mbf/fx;

    // 将特征点分配到图像网格中
    AssignFeaturesToGrid();

    std::cout<< "       ::视觉构造耗时:" << visual_timer.toc();

    loopthread.join();

    // calibration
    pcl::transformPointCloud(corner_points_sharp_, corner_points_sharp_, extrinsicMatrix);
    pcl::transformPointCloud(surface_points_flat_, surface_points_flat_, extrinsicMatrix);
    pcl::transformPointCloud(corner_points_less_sharp_, corner_points_less_sharp_, extrinsicMatrix);
    pcl::transformPointCloud(surface_points_less_flat_, surface_points_less_flat_, extrinsicMatrix);
    pcl::transformPointCloud(surface_points_flat_normal_, surface_points_flat_normal_, extrinsicMatrix);
    pcl::transformPointCloud(surface_points_less_flat_normal_, surface_points_less_flat_normal_, extrinsicMatrix);

    std::cout<< "         单帧构造总耗时:" << sum_timer.toc() << std::endl;

}


void Frame::CalculateRingAndTime(const PointICloud &all_cloud_in, PointIRTCloud &all_cloud_out) {

  //激光雷达线数初始化为64
  int N_SCANS = 64;
  //扫描周期, velodyne频率10Hz，周期0.1s
  const double scanPeriod = 0.1;

  size_t point_size = all_cloud_in.size();

  //每次扫描是一条线，看作者的数据集激光x向前，y向左，那么下面就是线一端到另一端
  //atan2的输出为-pi到pi(PS:atan输出为-pi/2到pi/2)
  //计算旋转角时取负号是因为velodyne是顺时针旋转
  float startOri = -atan2(all_cloud_in.points[0].y, all_cloud_in.points[0].x)+ M_PI;
  float endOri = -atan2(all_cloud_in.points[point_size - 1].y,
                        all_cloud_in.points[point_size - 1].x) + 3 * M_PI;

  //激光间距收束到1pi到3pi
  if (endOri - startOri > 3 * M_PI)
  {
      endOri -= 2 * M_PI;
  }
  else if (endOri - startOri < M_PI)
  {
      endOri += 2 * M_PI;
  }
//  printf("start Ori %fPI\nend Ori %fPI\n", startOri/M_PI, endOri/M_PI);

  //记录总点数
  int count = point_size;

  //按线数保存的点云集合
  std::vector<pcl::PointCloud<PointIRT>> laserCloudScans(N_SCANS);

  PointIRT p;
  double max_angle=-10000, min_angle=10000;
  for (size_t i = 0; i < point_size; ++i) {
    p.x = all_cloud_in.points[i].x;
    p.y = all_cloud_in.points[i].y;
    p.z = all_cloud_in.points[i].z;
    p.intensity = all_cloud_in.points[i].intensity;
    p.timestamp = 0.1;

    //求仰角atan输出为-pi/2到pi/2，实际看scanID应该每条线之间差距是2度
    float angle = atan(p.z / sqrt(p.x * p.x + p.y * p.y)) * 180 / M_PI;
    int scanID = -1;
    if(angle<min_angle) min_angle=angle;
    if(angle>max_angle) max_angle=angle;


    if (angle >= -8.83)
        scanID = int((2 - angle) * 3.0 + 0.5);
    else
        scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);
//        printf("angle %f scanID %d \n", angle, scanID);

    if (scanID > (N_SCANS - 1) || scanID < 0)
    {
      count--;
      continue;
    }

    p.ring = scanID;
    //按线分类保存
    laserCloudScans[scanID].push_back(p);
    all_cloud_out.points.push_back(p);
  }

  point_size = count;
  std::cout << "[CalculateRingAndTime]::总点数:" << all_cloud_in.points.size()
            << "       虚拟线束化点数:" << point_size
            << "    max angle:" << max_angle
            << "    min angle:" << min_angle << std::endl;
}


void Frame::PointToImage(const PointIRTCloud &all_cloud_in) {
  auto &points = all_cloud_in.points;
  size_t all_cloud_size = points.size();
  float startOri = -atan2(points[0].y, points[0].x) + M_PI;
  float endOri = -atan2(points[all_cloud_size - 1].y,
                        points[all_cloud_size - 1].x) + 3 * M_PI;
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

  bool halfPassed = false;
  std::vector<int> index_num(lidarconfig_->sensor_type, 0);    // 每一行多少点
  int ground_pointnum=0;

  for (size_t i = 0; i < all_cloud_size; ++i) {

    float azi_rad = -atan2(points[i].y, points[i].x);
    if (!halfPassed) {
      if (azi_rad < startOri - M_PI / 2) {
        azi_rad += 2 * M_PI;
      } else if (azi_rad > startOri + M_PI * 3 / 2) {
        azi_rad -= 2 * M_PI;
      }
      if (azi_rad - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      azi_rad += 2 * M_PI;
      if (azi_rad < endOri - M_PI * 3 / 2) {
        azi_rad += 2 * M_PI;
      } else if (azi_rad > endOri + M_PI / 2) {
        azi_rad -= 2 * M_PI;
      }
    }

    float azi_rad_rel = azi_rad - startOri;

    int col = std::fmod(int(azi_rad_rel / (2 * M_PI) * lidarconfig_->col_num_) + lidarconfig_->col_num_, lidarconfig_->col_num_);
    int row = points[i].ring;

    if (col < 0 || col >= lidarconfig_->col_num_ || row < 0 || row >= lidarconfig_->row_num_) {
      continue;
    }

    // -3=没有占用  -1=地面点   0=非地面点
    if (range_image_[row][col].point_state == -3) {
      if (points[i].z < lidarconfig_->ground_z_bound) {
        range_image_[row][col].point_state = -1;
        ground_pointnum++;
      } else {
        range_image_[row][col].point_state = 0;
      }
      range_image_[row][col].index = index_num[row];
      laser_scans_[row]->push_back(points[i]);
      ImageIndex index_temp;
      index_temp.row = row;
      index_temp.col = col;
      image_index_[row].push_back(index_temp);
      ++index_num[row];

      range_img_visual_.at<double>(row,col)=mathutils::CalcPointDistance(points[i]);

    } else {
      //如果已经占用选择距离更近的那个
      int scan_index = range_image_[row][col].index;
      float point_dis_1 = mathutils::CalcPointDistance(points[i]);
      float point_dis_2 = mathutils::CalcPointDistance(laser_scans_[row]->points[scan_index]);
      if (point_dis_1 < point_dis_2) {
        if (points[i].z < lidarconfig_->ground_z_bound) {
          range_image_[row][col].point_state = -1;
          ground_pointnum++;
        } else {
          range_image_[row][col].point_state = 0;
        }
        laser_scans_[row]->points[scan_index] = points[i];
        range_img_visual_.at<double>(row,col)=mathutils::CalcPointDistance(points[i]);
      }
    }
  }

  size_t cloud_size = 0;
  for (int i = 0; i < lidarconfig_->row_num_; i++) {
    cloud_in_rings_ += (*laser_scans_[i]);
    IndexRange range(cloud_size, 0);
    cloud_size += (*laser_scans_[i]).size();
    range.second = (cloud_size > 0 ? cloud_size - 1 : 0);
    scan_ranges_.push_back(range);
    // 比如说第一行18个点  第二行19个点  scan_ranges_[0]就是（0.17）  scan_ranges_[2]=（18,18+19）
  }

  std::cout << "[PointToImage]::深度图化点数：" << cloud_size
            << "        地面点数：" << ground_pointnum << std::endl;
}



// 提取角点前的准备，主要是设置这个掩膜，根据同一线点和点的距离关系来设置，设置了掩膜（1）的就不做后面的判断了
void Frame::PrepareRing_corner(const PointIRTCloud &scan) {

  size_t scan_size = scan.size();
  //预处理掩膜
  scan_ring_mask_.resize(scan_size);
  scan_ring_mask_.assign(scan_size, 0);
  // // 记录每个scan的结束index，忽略后n个点，开始和结束处的点云容易产生不闭合的“接缝”，对提取edge feature不利
  for (size_t i = 0 + lidarconfig_->num_curvature_regions_corner; i + lidarconfig_->num_curvature_regions_corner < scan_size; ++i) {
    const PointIRT &p_prev = scan[i - 1];
    const PointIRT &p_curr = scan[i];
    const PointIRT &p_next = scan[i + 1];

    float diff_next2 = mathutils::CalcSquaredDiff(p_curr, p_next);

    // about 30 cm 如果和下一个点的距离超过30cm
    if (diff_next2 > 0.1) {
      float depth = mathutils::CalcPointDistance(p_curr);
      float depth_next = mathutils::CalcPointDistance(p_next);

      // 比较深度
      if (depth > depth_next) {
        // to closer point
        float weighted_diff = sqrt(mathutils::CalcSquaredDiff(p_next, p_curr, depth_next / depth)) / depth_next;
        // relative distance
        if (weighted_diff < 0.1) {
          // 把上num_curvature_regions_corner个点到当前点掩膜置位
          fill_n(&scan_ring_mask_[i - lidarconfig_->num_curvature_regions_corner], lidarconfig_->num_curvature_regions_corner + 1, 1);
          continue;
        }
      } else {
        float weighted_diff = sqrt(mathutils::CalcSquaredDiff(p_curr, p_next, depth / depth_next)) / depth;
        if (weighted_diff < 0.1) {
          // 把下num_curvature_regions_corner个点置位
          fill_n(&scan_ring_mask_[i + 1], lidarconfig_->num_curvature_regions_corner, 1);
          continue;
        }
      }
    }

    float diff_prev2 = mathutils::CalcSquaredDiff(p_curr, p_prev);
    float dis2 = mathutils::CalcSquaredPointDistance(p_curr);

    // for this point -- 1m -- 1.5cm
    if (diff_next2 > 0.0002 * dis2 && diff_prev2 > 0.0002 * dis2) {
      scan_ring_mask_[i - 0] = 1;
    }
  }

//  for(int i=0; i<scan_size; i++){
//    if(scan_ring_mask_[1]==1){
//      mask_points_.push_back(scan[i]);
//    }
//  }

}


void Frame::PrepareRing_flat(const PointIRTCloud &scan) {

  size_t scan_size = scan.size();
  scan_ring_mask_.resize(scan_size);
  scan_ring_mask_.assign(scan_size, 0);
  for (size_t i = 0 + lidarconfig_->num_curvature_regions_flat; i + lidarconfig_->num_curvature_regions_flat < scan_size; ++i) {
    const PointIRT &p_prev = scan[i - 1];
    const PointIRT &p_curr = scan[i];
    const PointIRT &p_next = scan[i + 1];

    float diff_next2 = mathutils::CalcSquaredDiff(p_curr, p_next);

    // about 30 cm
    if (diff_next2 > 0.1) {
      float depth = mathutils::CalcPointDistance(p_curr);
      float depth_next = mathutils::CalcPointDistance(p_next);

      if (depth > depth_next) {
        // to closer point
        // 是区分两个点到激光雷达的向量  基本保持一条直线
        float weighted_diff = sqrt(mathutils::CalcSquaredDiff(p_next, p_curr, depth_next / depth)) / depth_next;
        // relative distance
        if (weighted_diff < 0.1) {
          fill_n(&scan_ring_mask_[i - lidarconfig_->num_curvature_regions_flat], lidarconfig_->num_curvature_regions_flat + 1, 1);
          continue;
        }
      } else {
        float weighted_diff = sqrt(mathutils::CalcSquaredDiff(p_curr, p_next, depth / depth_next)) / depth;
        if (weighted_diff < 0.1) {
          fill_n(&scan_ring_mask_[i + 1], lidarconfig_->num_curvature_regions_flat, 1);
          continue;
        }
      }
    }

//     float diff_prev2 = CalcSquaredDiff(p_curr, p_prev);
//     float dis2 = CalcSquaredPointDistance(p_curr);

//     // for this point -- 1m -- 1.5cm
//     if (diff_next2 > 0.0002 * dis2 && diff_prev2 > 0.0002 * dis2) {
//       scan_ring_mask_[i - 0] = 1;
//     }
  }
//  for(int i=0; i<scan_size; i++){
//    if(scan_ring_mask_[1]==1){
//      mask_points_.push_back(scan[i]);
//    }
//  }
}


/// 算曲率
void Frame::PrepareSubregion_corner(const PointIRTCloud &scan, const size_t idx_start, const size_t idx_end) {

//  cout << ">>>>>>> " << idx_ring << ", " << idx_start << ", " << idx_end << " <<<<<<<" << endl;
//  const PointIRTCloud &scan = laser_scans_[idx_ring];
  size_t region_size = idx_end - idx_start + 1;
  curvature_idx_pairs_.clear();
  curvature_idx_pairs_.resize(region_size);

  // 算曲率  邻域是左右多少点  LOAM中的公式，那个曲率其实就代表弯曲的模长
  // https://blog.csdn.net/shoufei403/article/details/103664877
  for (size_t i = idx_start, in_region_idx = 0; i <= idx_end; ++i, ++in_region_idx) {

    int num_point_neighbors = 2 * lidarconfig_->num_curvature_regions_corner;
    float diff_x = -num_point_neighbors * scan[i].x;
    float diff_y = -num_point_neighbors * scan[i].y;
    float diff_z = -num_point_neighbors * scan[i].z;

    for (int j = 1; j <= lidarconfig_->num_curvature_regions_corner; ++j) {
      diff_x += scan[i + j].x + scan[i - j].x;
      diff_y += scan[i + j].y + scan[i - j].y;
      diff_z += scan[i + j].z + scan[i - j].z;
    }


    float curvature = (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)/num_point_neighbors;
    pair<float, size_t> curvature_idx_(curvature, i);
    curvature_idx_pairs_[in_region_idx] = curvature_idx_;
//    _regionCurvature[regionIdx] = diffX * diffX + diffY * diffY + diffZ * diffZ;
//    _regionSortIndices[regionIdx] = i;
  }

  sort(curvature_idx_pairs_.begin(), curvature_idx_pairs_.end());
/*
  for (const auto &pair : curvature_idx_pairs_) {
    cout << pair.first << " " << pair.second << endl;
  }
*/
}


// 算曲率
void Frame::PrepareSubregion_flat(const PointIRTCloud &scan, const size_t idx_start, const size_t idx_end) {

//  cout << ">>>>>>> " << idx_ring << ", " << idx_start << ", " << idx_end << " <<<<<<<" << endl;
//  const PointIRTCloud &scan = laser_scans_[idx_ring];
  size_t region_size = idx_end - idx_start + 1;
  size_t scan_size = scan.size();
  curvature_idx_pairs_.clear();
  curvature_idx_pairs_.resize(region_size);

  for (size_t i = idx_start, in_region_idx = 0; i <= idx_end; ++i, ++in_region_idx) {

    float point_dist = mathutils::CalcPointDistance(scan[i]);
    int num_curvature_regions = int(25.0 / point_dist + 0.5) + 1;

    if (i < num_curvature_regions || i + num_curvature_regions >= scan_size) {
      num_curvature_regions = lidarconfig_->num_curvature_regions_flat;
    }

    int num_point_neighbors = 2 * num_curvature_regions;
    float diff_x = -num_point_neighbors * scan[i].x;
    float diff_y = -num_point_neighbors * scan[i].y;
    float diff_z = -num_point_neighbors * scan[i].z;

    for (int j = 1; j <= num_curvature_regions; ++j) {
      diff_x += scan[i + j].x + scan[i - j].x;
      diff_y += scan[i + j].y + scan[i - j].y;
      diff_z += scan[i + j].z + scan[i - j].z;
    }

    float curvature = (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)/num_point_neighbors;
    pair<float, size_t> curvature_idx_(curvature, i);
    curvature_idx_pairs_[in_region_idx] = curvature_idx_;
  }
  sort(curvature_idx_pairs_.begin(), curvature_idx_pairs_.end());
}




void Frame::ExtractFeaturePoints() {
  int unstable_pointnum=0;
  int labelCount = 1;
  vector<Eigen::Vector3d> surface_points_normal_temp;
  ///< i is #ring, j is #subregion, k is # in region
  // 一线一线去处理
  for (size_t i = 0; i < lidarconfig_->row_num_; ++i) {

    size_t start_idx = scan_ranges_[i].first;
    size_t end_idx = scan_ranges_[i].second;

    // skip too short scans
    if (lidarconfig_->num_curvature_regions_corner < lidarconfig_->num_curvature_regions_flat) {
      if (end_idx <= start_idx + 2 * lidarconfig_->num_curvature_regions_flat) {
        continue;
      }
    } else {
      if (end_idx <= start_idx + 2 * lidarconfig_->num_curvature_regions_corner) {
        continue;
      }
    }

    PointIRTCloud &scan_ring = *laser_scans_[i];
    const vector<ImageIndex> &index_ring = image_index_[i];
    size_t scan_size = scan_ring.size();

    // 提取角点
    if (lidarconfig_->using_sharp_point &&
        lidarconfig_->lower_ring_num_sharp_point <= (i + 1) && i < lidarconfig_->upper_ring_num_sharp_point) {

      // 设置掩膜
      PrepareRing_corner(scan_ring);

      // 分区域提取
      for (int j = 0; j < lidarconfig_->num_scan_subregions; ++j) {
        // 算子区域的索引
        // ((s+d)*N+j*(e-s-d))/N, ((s+d)*N+(j+1)*(e-s-d))/N-1
        size_t sp = ((0 + lidarconfig_->num_curvature_regions_corner) * (lidarconfig_->num_scan_subregions - j)
            + (scan_size - lidarconfig_->num_curvature_regions_corner) * j) / lidarconfig_->num_scan_subregions;
        size_t ep = ((0 + lidarconfig_->num_curvature_regions_corner) * (lidarconfig_->num_scan_subregions - 1 - j)
            + (scan_size - lidarconfig_->num_curvature_regions_corner) * (j + 1)) / lidarconfig_->num_scan_subregions - 1;

        // skip empty regions
        if (ep <= sp) {
          continue;
        }

        size_t region_size = ep - sp + 1;

        // extract corner features
        PrepareSubregion_corner(scan_ring, sp, ep);

        int num_largest_picked = 0;
        // 倒序因为要从大曲率的开始
        for (size_t k = region_size; k > 0; --k) {
          // k must be greater than 0
          const pair<float, size_t> &curvature_idx = curvature_idx_pairs_[k - 1];
          float curvature = curvature_idx.first;
          size_t idx = curvature_idx.second;
          size_t in_scan_idx = idx - 0; // scan start index is 0 for all ring scans
          size_t in_region_idx = idx - sp;

          // 标记为地面点的不进行下一步
          if (range_image_[i][image_index_[i][in_scan_idx].col].point_state == -1) {
            continue;
          }

          // 掩膜没有被设置才会进入下一步  并且要符合曲率阈值
          if (scan_ring_mask_[in_scan_idx] == 0 && curvature > lidarconfig_->sharp_curv_th) {

            vector<ImageIndex> queue_ind;
            ImageIndex index = image_index_[i][in_scan_idx];

            // 对满足曲率的点进行角度阈值聚类  角度越大，两个点离得越近
            if (range_image_[i][index.col].point_state == 0) {

              float d1, d2, alpha, angle;
              int fromIndX, fromIndY, thisIndX, thisIndY;
              bool lineCountFlag[lidarconfig_->row_num_] = {false};

              // vector<ImageIndex> queue_ind;
              queue_ind.push_back(image_index_[i][in_scan_idx]);
              int queueSize = 1;
              int queueStartInd = 0;

              while (queueSize > 0) {
                // Pop point
                fromIndX = queue_ind[queueStartInd].row;
                fromIndY = queue_ind[queueStartInd].col;
                --queueSize;
                ++queueStartInd;
                // Mark popped point
                range_image_[fromIndX][fromIndY].point_state = labelCount;

                // 深度图上该点周围八个点
                vector<ImageIndex> neighbor;
                neighbor.push_back(ImageIndex{-1, 0});
                neighbor.push_back(ImageIndex{1, 0});
                neighbor.push_back(ImageIndex{0, -1});
                neighbor.push_back(ImageIndex{0, 1});

                neighbor.push_back(ImageIndex{-1, -1});
                neighbor.push_back(ImageIndex{1, -1});
                neighbor.push_back(ImageIndex{0, -1});
                neighbor.push_back(ImageIndex{0, 1});

                // Loop through all the neighboring grids of popped grid
                for (auto iter = neighbor.begin(); iter != neighbor.end(); ++iter) {
                  // new index
                  thisIndX = fromIndX + (*iter).row;
                  thisIndY = fromIndY + (*iter).col;
                  // index should be within the boundary
                  if (thisIndX < 0 || thisIndX >= lidarconfig_->row_num_) {
                    continue;
                  }
                  // at range image margin (left or right side)
                  // if (thisIndY < 0 || thisIndY >= col_num_)
                  //   continue;
                  if (thisIndY < 0) {
                    thisIndY = lidarconfig_->col_num_ - 1;
                  }
                  if (thisIndY >= lidarconfig_->col_num_) {
                    thisIndY = 0;
                  }
                  // prevent infinite loop (caused by put already examined point back)
                  if (range_image_[thisIndX][thisIndY].point_state != 0) {
                    continue;
                  }

                  PointIRT p1 = laser_scans_[fromIndX]->points[range_image_[fromIndX][fromIndY].index];
                  PointIRT p2 = laser_scans_[thisIndX]->points[range_image_[thisIndX][thisIndY].index];

                  Eigen::Vector3d vector_p1{p1.x, p1.y, p1.z};
                  Eigen::Vector3d vector_p2{p2.x, p2.y, p2.z};

                  d1 = std::max(vector_p1.norm(), vector_p2.norm());
                  d2 = std::min(vector_p1.norm(), vector_p2.norm());

                  alpha = acos((vector_p1.dot(vector_p2)) / (vector_p1.norm() * vector_p2.norm()));
                  angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

                  if (angle > 1) {
                    queue_ind.push_back(ImageIndex{thisIndX, thisIndY});
                    ++queueSize;

                    range_image_[thisIndX][thisIndY].point_state = labelCount;
                    lineCountFlag[thisIndX] = true;
                  }
                }
              }

              // check if this segment is valid
              bool feasibleSegment = false;
              int lineCount = 0;
              for (int r = 0; r < lidarconfig_->row_num_; ++r) {
                if (lineCountFlag[r] == true) {
                  ++lineCount;
                }
              }
              if (lineCount >= 3) {
                feasibleSegment = true;
              }

              if (feasibleSegment == true) {
                ++labelCount;
              } else {
                for (int n = 0; n < queueStartInd; ++n) {
                  // -2表示聚类不稳定的类
                  range_image_[queue_ind[n].row][queue_ind[n].col].point_state = -2;
                  unstable_pointnum++;
                }
              }
            }

            if (range_image_[i][image_index_[i][in_scan_idx].col].point_state < 1) {
              continue;
            }


            if (num_largest_picked < lidarconfig_->max_corner_less_sharp) {
              ImageIndex index = image_index_[i][in_scan_idx];
              for (int a = 0 ; a < queue_ind.size(); ++a) {
                laser_scans_[queue_ind[a].row]->points[range_image_[queue_ind[a].row][queue_ind[a].col].index].intensity = range_image_[queue_ind[a].row][queue_ind[a].col].point_state;
              }

              if (range_image_[i][index.col].point_state > less_sharp_.size()) {
                less_sharp_.push_back(queue_ind);
              }

              corner_points_less_sharp_.push_back(scan_ring[in_scan_idx]);
              if (num_largest_picked < lidarconfig_->max_corner_sharp) {
                corner_points_sharp_.push_back(scan_ring[in_scan_idx]);
                range_image_[i][index.col].feature_state = 1;
              }
//              MaskPickedInRing(scan_ring, in_scan_idx);
              ++num_largest_picked;
            }

            if (num_largest_picked >= lidarconfig_->max_corner_less_sharp) {
              break;
            }
          }
        }
      } /// j
    }

    // 如果线束在设置的这个范围内  就提取平面点
    if (lidarconfig_->using_flat_point &&
        ( (lidarconfig_->lower_ring_num_x_rot <= (i + 1) && (i + 1) <= lidarconfig_->upper_ring_num_x_rot) ||
          (lidarconfig_->lower_ring_num_y_rot <= (i + 1) && (i + 1) <= lidarconfig_->upper_ring_num_y_rot) ||
          (lidarconfig_->lower_ring_num_z_trans <= (i + 1) && (i + 1) <= lidarconfig_->upper_ring_num_z_trans) ||
          (lidarconfig_->lower_ring_num_z_rot_xy_trans <= (i + 1) && (i + 1) <= lidarconfig_->lower_ring_num_z_rot_xy_trans) )) {

      // 掩膜
      PrepareRing_flat(scan_ring);

      // extract features from equally sized scan regions
      for (int j = 0; j < lidarconfig_->num_scan_subregions; ++j) {
        // ((s+d)*N+j*(e-s-d))/N, ((s+d)*N+(j+1)*(e-s-d))/N-1
        size_t sp = ((0 + lidarconfig_->num_curvature_regions_flat) * (lidarconfig_->num_scan_subregions - j)
            + (scan_size - lidarconfig_->num_curvature_regions_flat) * j) / lidarconfig_->num_scan_subregions;
        size_t ep = ((0 + lidarconfig_->num_curvature_regions_flat) * (lidarconfig_->num_scan_subregions - 1 - j)
            + (scan_size - lidarconfig_->num_curvature_regions_flat) * (j + 1)) / lidarconfig_->num_scan_subregions - 1;

        // skip empty regions
        if (ep <= sp) {
          continue;
        }
        size_t region_size = ep - sp + 1;

        // 曲率计算 邻域按照平面点设置的来计算
        // extract flat surface features
        PrepareSubregion_flat(scan_ring, sp, ep);

        int num_largest_picked = 0;
        // 正序因为要从小曲率的开始
        for (size_t k = 1; k <= region_size; ++k) {
          const pair<float, size_t> &curvature_idx = curvature_idx_pairs_[k - 1];
          float curvature = curvature_idx.first;

          size_t idx = curvature_idx.second;
          size_t in_scan_idx = idx - 0; // scan start index is 0 for all ring scans
          size_t in_region_idx = idx - sp;

          // 满足平面阈值要求
          if (scan_ring_mask_[in_scan_idx] == 0 && curvature < lidarconfig_->surf_curv_th) {
            float point_dist = mathutils::CalcPointDistance(scan_ring[in_scan_idx]);
            // 算了一个邻域大小 不知道是什么原理
            int num_curvature_regions = (point_dist * 0.01 / (mathutils::DegToRad(lidarconfig_->deg_diff) * point_dist) + 0.5);

            ImageIndex index = image_index_[i][in_scan_idx];
            PointIRTCloud search_points;

            // 把这一线邻域内的点都加进去
            search_points.push_back(scan_ring[in_scan_idx]);
            for (int c = 1; c <= num_curvature_regions; ++c) {
              if (index.col + c < lidarconfig_->col_num_) {
                if (range_image_[i][index.col + c].point_state != -3) {
                  search_points.push_back(scan_ring[range_image_[i][index.col + c].index]);
                }
              }
              if (index.col >= c) {
                if (range_image_[i][index.col - c].point_state != -3) {
                  search_points.push_back(scan_ring[range_image_[i][index.col - c].index]);
                }
              }
            }

            // 把上一线邻域内的点都加进去
            if (i > 0) {
              if (range_image_[i - 1][index.col].point_state != -3) {
                search_points.push_back(laser_scans_[i - 1]->points[range_image_[i - 1][index.col].index]);
              }
              for (int c = 1; c <= num_curvature_regions; ++c) {
                if (index.col + c < lidarconfig_->col_num_) {
                  if (range_image_[i - 1][index.col + c].point_state != -3) {
                    search_points.push_back(laser_scans_[i - 1]->points[range_image_[i - 1][index.col + c].index]);
                  }
                }
                if (index.col >= c) {
                  if (range_image_[i - 1][index.col - c].point_state != -3) {
                    search_points.push_back(laser_scans_[i - 1]->points[range_image_[i - 1][index.col - c].index]);
                  }
                }
              }
            }

            // 把下一线邻域内的点都加进去
            if (i + 1 < lidarconfig_->row_num_) {
              if (range_image_[i + 1][index.col].point_state != -3) {
                search_points.push_back(laser_scans_[i + 1]->points[range_image_[i + 1][index.col].index]);
              }
              for (int c = 1; c <= num_curvature_regions; ++c) {
                if (index.col + c < lidarconfig_->col_num_) {
                  if (range_image_[i + 1][index.col + c].point_state != -3) {
                    search_points.push_back(laser_scans_[i + 1]->points[range_image_[i + 1][index.col + c].index]);
                  }
                }
                if (index.col >= c) {
                  if (range_image_[i + 1][index.col - c].point_state != -3) {
                    search_points.push_back(laser_scans_[i + 1]->points[range_image_[i + 1][index.col - c].index]);
                  }
                }
              }
            }

            size_t search_points_num = search_points.size();
            if (search_points_num < 5) {
              continue;
            }

            // 从上面的三线邻域点中选了1m之内的点
            vector<Eigen::Vector3d> near_point;
            for (size_t s = 0; s < search_points_num; ++s) {
              double dis = mathutils::CalcSquaredDiff(scan_ring[in_scan_idx], search_points[s]);
              if (dis < lidarconfig_->max_sq_dis) {
                Eigen::Vector3d tmp(search_points[s].x,
                                    search_points[s].y,
                                    search_points[s].z);
                near_point.push_back(tmp);
              }
            }

//            // 协方差矩阵
//            Eigen::Vector3d center(0, 0, 0);
//            for (int s = 0; s < near_point.size(); ++s) {
//              center += near_point[s];
//            }
//            center = center / near_point.size();
//            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
//            for (int s = 0; s < near_point.size(); ++s) {
//              Eigen::Matrix<double, 3, 1> tmpZeroMean = near_point[s] - center;
//              covMat += (tmpZeroMean * tmpZeroMean.transpose());
//            }

//            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

//            // note Eigen library sort eigenvalues in increasing order
//            Eigen::Vector3d unit_direction = saes.eigenvectors().col(0);

            // if (saes.eigenvalues()[1] < 10 * saes.eigenvalues()[0]) {
            //   continue;
            // }

            // 算法线
            Eigen::MatrixXd matA0;
            Eigen::MatrixXd matB0;
            matA0.resize(near_point.size(), 3);
            matB0.resize(near_point.size(), 1);
            for (int s = 0; s < near_point.size(); ++s)
            {
              matA0(s, 0) = near_point[s].x();
              matA0(s, 1) = near_point[s].y();
              matA0(s, 2) = near_point[s].z();
              matB0(s, 0) = -1.0;
            }

            Eigen::Vector3d point_normal = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / point_normal.norm();
            point_normal.normalize();

            bool planeValid = true;
            for (int s = 0; s < near_point.size(); ++s)
            {
              if (fabs( point_normal(0) * near_point[s].x() +
                        point_normal(1) * near_point[s].y() +
                        point_normal(2) * near_point[s].z() + negative_OA_dot_norm) > 0.1)
              {
                planeValid = false;
                break;
              }
            }

            if (planeValid == false) {
              continue;
            }

            PointIRT normal;
            normal.x = point_normal.x();
            normal.y = point_normal.y();
            normal.z = point_normal.z();

            if (num_largest_picked < lidarconfig_->max_surf_less_flat) {
              surface_points_less_flat_.push_back(scan_ring[in_scan_idx]);
              surface_points_less_flat_index_.push_back(index_ring[in_scan_idx]);
              surface_points_less_flat_normal_.push_back(normal);
              surface_points_normal_temp.push_back(point_normal);
              if (num_largest_picked < lidarconfig_->max_surf_flat) {
                surface_points_flat_.push_back(scan_ring[in_scan_idx]);
                surface_points_flat_normal_.push_back(normal);
              }
//              MaskPickedInRing(scan_ring, in_scan_idx);
              ++num_largest_picked;
            }

            if (num_largest_picked >= lidarconfig_->max_surf_less_flat) {
              break;
            }
          }
        }
      } /// j
    }
  } /// i
  // 处理完毕

} // ExtractFeaturePoints




void Frame::lidarProcess(){
  TicToc lidar_timer;

  vector<int> ind;
  lidar_inputPtr_->is_dense = false;
  pcl::removeNaNFromPointCloud(*lidar_inputPtr_, *lidar_inputPtr_, ind);
  PointIRTCloud laser_cloud_out;
  CalculateRingAndTime(*lidar_inputPtr_, laser_cloud_out);
  PointToImage(laser_cloud_out);
  ExtractFeaturePoints();
  std::cout << "         lidar time:" << lidar_timer.toc();
}

cv::Mat Frame::getDepthimg(){ return Depthimg_; }

cv::Mat Frame::getRangeImage(){ return range_img_visual_; }
PointIRTCloud Frame::getCornerPoints(){ return corner_points_sharp_; }
PointIRTCloud Frame::getFlatPoints(){ return surface_points_flat_; }
PointIRTCloud Frame::getLessCornerPoints(){ return corner_points_less_sharp_; }
PointIRTCloud Frame::getLessFlatPoints(){ return surface_points_less_flat_; }

/**
 * @brief 将提取的ORB特征点分配到图像网格中
 * 
 */
void Frame::AssignFeaturesToGrid()
{
    // Step 1  给存储特征点的网格数组 Frame::mGrid 预分配空间
    // ? 这里0.5 是为什么？节省空间？
    // FRAME_GRID_COLS = 64，FRAME_GRID_ROWS=48
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    // 开始对mGrid这个二维数组中的每一个vector元素遍历并预分配空间
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    // Step 2 遍历每个特征点，将每个特征点在mvKeysUn中的索引值放到对应的网格mGrid中
    for(int i=0;i<N;i++)
    {
        // 从类的成员变量中获取已经去畸变后的特征点
        const cv::KeyPoint &kp = mvKeysUn[i];

        // 存储某个特征点所在网格的网格坐标，nGridPosX范围：[0,FRAME_GRID_COLS], nGridPosY范围：[0,FRAME_GRID_ROWS]
        int nGridPosX, nGridPosY;
        // 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            // 如果找到特征点所在网格坐标，将这个特征点的索引添加到对应网格的数组mGrid中
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}


/**
 * @brief 提取图像的ORB特征点，提取的关键点存放在mvKeys，描述子存放在mDescriptors
 * 
 * @param[in] flag          标记是左图还是右图。0：左图  1：右图
 * @param[in] im            等待提取特征点的图像
 */
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    // 判断是左图还是右图
    if(flag==0)
        // 左图的话就套使用左图指定的特征点提取器，并将提取结果保存到对应的变量中 
        // 这里使用了仿函数来完成，重载了括号运算符 ORBextractor::operator() 
        (*mpORBextractorLeft)(im,                //待提取特征点的图像
                              mvKeys,            //输出变量，用于保存提取后的特征点
                              mDescriptors);     //输出变量，用于保存特征点的描述子
    else
        // 右图的话就需要使用右图指定的特征点提取器，并将提取结果保存到对应的变量中 
        (*mpORBextractorRight)(im, mvKeysRight, mDescriptorsRight);
}


// 设置相机姿态
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}


//根据Tcw计算mRcw、mtcw和mRwc、mOw
void Frame::UpdatePoseMatrices()
{
    // mOw：    当前相机光心在世界坐标系下坐标
    // mTcw：   世界坐标系到相机坐标系的变换矩阵
    // mRcw：   世界坐标系到相机坐标系的旋转矩阵
    // mtcw：   世界坐标系到相机坐标系的平移向量
    // mRwc：   相机坐标系到世界坐标系的旋转矩阵

    //从变换矩阵中提取出旋转矩阵
    //注意，rowRange这个只取到范围的左边界，而不取右边界
    mRcw = mTcw.rowRange(0,3).colRange(0,3);

    // mRcw求逆即可
    mRwc = mRcw.t();

    // 从变换矩阵中提取出旋转矩阵
    mtcw = mTcw.rowRange(0,3).col(3);

    // mTcw 求逆后是当前相机坐标系变换到世界坐标系下，对应的光心变换到世界坐标系下就是 mTcw的逆 中对应的平移向量
    mOw = -mRcw.t()*mtcw;
}


/**
 * @brief 判断路标点是否在视野中
 * 步骤
 * Step 1 获得这个地图点的世界坐标
 * Step 2 关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，返回false
 * Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
 * Step 4 关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
 * Step 5 关卡四：计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值, 若小于设定阈值，返回false
 * Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
 * Step 7 记录计算得到的一些参数
 * @param[in] pMP                       当前地图点
 * @param[in] viewingCosLimit           夹角余弦，用于限制地图点和光心连线和法线的夹角
 * @return true                         地图点合格，且在视野内
 * @return false                        地图点不合格，抛弃
 */
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    // mbTrackInView是决定一个地图点是否进行重投影的标志
    // 这个标志的确定要经过多个函数的确定，isInFrustum()只是其中的一个验证关卡。这里默认设置为否
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    // Step 1 获得这个地图点的世界坐标
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    // 根据当前帧(粗糙)位姿转化到当前相机坐标系下的三维点Pc
    const cv::Mat Pc = mRcw*P+mtcw; 
    const float &PcX = Pc.at<float>(0);
    const float &PcY = Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    // Step 2 关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，直接返回false
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    // Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
    const float invz = 1.0f/PcZ;			
    const float u=fx*PcX*invz+cx;			
    const float v=fy*PcY*invz+cy;			

    // 判断是否在图像边界中，只要不在那么就说明无法在当前帧下进行重投影
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    // Step 4 关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
    // 得到认为的可靠距离范围:[0.8f*mfMinDistance, 1.2f*mfMaxDistance]
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();

    // 得到当前地图点距离当前帧相机光心的距离,注意P，mOw都是在同一坐标系下才可以
    // mOw：当前相机光心在世界坐标系下坐标
    const cv::Mat PO = P-mOw;
    // 取模就得到了距离
    const float dist = cv::norm(PO);

    // 如果不在允许的尺度变化范围内，认为重投影不可靠
    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    // Step 5 关卡四：计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值, 若小于cos(viewingCosLimit), 即夹角大于viewingCosLimit弧度则返回
    cv::Mat Pn = pMP->GetNormal();

    // 计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值，注意平均观测方向为单位向量
    const float viewCos = PO.dot(Pn)/dist;

    //如果大于给定的阈值 cos(60°)=0.5，认为这个点方向太偏了，重投影不可靠，返回false
    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    // Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
    const int nPredictedLevel = pMP->PredictScale(dist,		//这个点到光心的距离
                                                  this);	//给出这个帧
    // Step 7 记录计算得到的一些参数
    // Data used by the tracking	
    // 通过置位标记 MapPoint::mbTrackInView 来表示这个地图点要被投影 
    pMP->mbTrackInView = true;	

    // 该地图点投影在当前图像（一般是左图）的像素横坐标
    pMP->mTrackProjX = u;	

    // bf/z其实是视差，相减得到右图（如有）中对应点的横坐标
    pMP->mTrackProjXR = u - mbf*invz; 

    // 该地图点投影在当前图像（一般是左图）的像素纵坐标
    pMP->mTrackProjY = v;

    // 根据地图点到光心距离，预测的该地图点的尺度层级
    pMP->mnTrackScaleLevel = nPredictedLevel;

    // 保存当前视角和法线夹角的余弦值
    pMP->mTrackViewCos = viewCos;

    // 执行到这里说明这个地图点在相机的视野中并且进行重投影是可靠的，返回true
    return true;
}


/**
 * @brief 找到在 以x,y为中心,半径为r的圆形内且金字塔层级在[minLevel, maxLevel]的特征点
 * 
 * @param[in] x                     特征点坐标x
 * @param[in] y                     特征点坐标y
 * @param[in] r                     搜索半径 
 * @param[in] minLevel              最小金字塔层级
 * @param[in] maxLevel              最大金字塔层级
 * @return vector<size_t>           返回搜索到的候选匹配点id
 */
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    // 存储搜索结果的vector
    vector<size_t> vIndices;
    vIndices.reserve(N);

    // Step 1 计算半径为r圆左右上下边界所在的网格列和行的id
    // 查找半径为r的圆左侧边界所在网格列坐标。这个地方有点绕，慢慢理解下：
    // (mnMaxX-mnMinX)/FRAME_GRID_COLS：表示列方向每个网格可以平均分得几个像素（肯定大于1）
    // mfGridElementWidthInv=FRAME_GRID_COLS/(mnMaxX-mnMinX) 是上面倒数，表示每个像素可以均分几个网格列（肯定小于1）
    // (x-mnMinX-r)，可以看做是从图像的左边界mnMinX到半径r的圆的左边界区域占的像素列数
    // 两者相乘，就是求出那个半径为r的圆的左侧边界在哪个网格列中
    // 保证nMinCellX 结果大于等于0
    const int nMinCellX = max(0, (int)floor( (x-mnMinX-r)*mfGridElementWidthInv));


    // 如果最终求得的圆的左边界所在的网格列超过了设定了上限，那么就说明计算出错，找不到符合要求的特征点，返回空vector
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    // 计算圆所在的右边界网格列索引
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1, (int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    // 如果计算出的圆右边界所在的网格不合法，说明该特征点不好，直接返回空vector
    if(nMaxCellX<0)
        return vIndices;

    // 后面的操作也都是类似的，计算出这个圆上下边界所在的网格行的id
    const int nMinCellY = max(0, (int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1, (int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    // 检查需要搜索的图像金字塔层数范围是否符合要求
    //? 疑似bug。(minLevel>0) 后面条件 (maxLevel>=0)肯定成立
    //? 改为 const bool bCheckLevels = (minLevel>=0) || (maxLevel>=0);
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    // Step 2 遍历圆形区域内的所有网格，寻找满足条件的候选特征点，并将其index放到输出里
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            // 获取这个网格内的所有特征点在 Frame::mvKeysUn 中的索引
            const vector<size_t> vCell = mGrid[ix][iy];
            // 如果这个网格中没有特征点，那么跳过这个网格继续下一个
            if(vCell.empty())
                continue;

            // 如果这个网格中有特征点，那么遍历这个图像网格中所有的特征点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                // 根据索引先读取这个特征点
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                // 保证给定的搜索金字塔层级范围合法
                if(bCheckLevels)
                {
                    // cv::KeyPoint::octave中表示的是从金字塔的哪一层提取的数据
                    // 保证特征点是在金字塔层级minLevel和maxLevel之间，不是的话跳过
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)		//? 为何特意又强调？感觉多此一举
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                // 通过检查，计算候选特征点到圆中心的距离，查看是否是在这个圆形区域之内
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                // 如果x方向和y方向的距离都在指定的半径之内，存储其index为候选特征点
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }
    return vIndices;
}


/**
 * @brief 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
 * 
 * @param[in] kp                    给定的特征点
 * @param[in & out] posX            特征点所在网格坐标的横坐标
 * @param[in & out] posY            特征点所在网格坐标的纵坐标
 * @return true                     如果找到特征点所在的网格坐标，返回true
 * @return false                    没找到返回false
 */
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    // 计算特征点x,y坐标落在哪个网格内，网格坐标为posX，posY
    // mfGridElementWidthInv=(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
    // mfGridElementHeightInv=(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    // Keypoint's coordinates are undistorted, which could cause to go out of the image
    // 因为特征点进行了去畸变，而且前面计算是round取整，所以有可能得到的点落在图像网格坐标外面
    // 如果网格坐标posX，posY超出了[0,FRAME_GRID_COLS] 和[0,FRAME_GRID_ROWS]，表示该特征点没有对应网格坐标，返回false
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    // 计算成功返回true
    return true;
}


/**
 * @brief 计算当前帧特征点对应的词袋Bow，主要是mBowVec 和 mFeatVec
 * 
 */
void Frame::ComputeBoW()
{
    // 判断是否以前已经计算过了，计算过了就跳过
    if(mBowVec.empty())
    {
        // 将描述子mDescriptors转换为DBOW要求的输入格式
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // 将特征点的描述子转换成词袋向量mBowVec以及特征向量mFeatVec
        mpORBvocabulary->transform(vCurrentDesc,	//当前的描述子vector
                                   mBowVec,			//输出，词袋向量，记录的是单词的id及其对应权重TF-IDF值
                                   mFeatVec,		//输出，记录node id及其对应的图像 feature对应的索引
                                   4);				//4表示从叶节点向前数的层数
    }
}


/**
 * @brief 用内参对特征点去畸变，结果报存在mvKeysUn中
 * 
 */
void Frame::UndistortKeyPoints()
{
    // Step 1 如果第一个畸变参数为0，不需要矫正。第一个畸变参数k1是最重要的，一般不为0，为0的话，说明畸变参数都是0
    //变量mDistCoef中存储了opencv指定格式的去畸变参数，格式为：(k1,k2,p1,p2,k3)
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }


    // Step 2 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    // Fill matrix with points
    // N为提取的特征点数量，为满足OpenCV函数输入要求，将N个特征点保存在N*2的矩阵中
    cv::Mat mat(N,2,CV_32F);
    // 遍历每个特征点，并将它们的坐标保存到矩阵中
    for(int i=0; i<N; i++)
    {
        //然后将这个特征点的横纵坐标分别保存
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // 函数reshape(int cn,int rows=0) 其中cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
    // 为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）
    mat=mat.reshape(2);
    cv::undistortPoints(mat,				//输入的特征点坐标
                        mat,				//输出的校正后的特征点坐标覆盖原矩阵
                        mK,					//相机的内参数矩阵
                        mDistCoef,	//相机畸变参数矩阵
                        cv::Mat(),	//一个空矩阵，对应为函数原型中的R
                        mK); 				//新内参数矩阵，对应为函数原型中的P

    //调整回只有一个通道，回归我们正常的处理方式
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    // Step 存储校正后的特征点
    mvKeysUn.resize(N);
    //遍历每一个特征点
    for(int i=0; i<N; i++)
    {
        //根据索引获取这个特征点
        //注意之所以这样做而不是直接重新声明一个特征点对象的目的是，能够得到源特征点对象的其他属性
        cv::KeyPoint kp = mvKeys[i];
        //读取校正后的坐标并覆盖老坐标
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}


/**
 * @brief 计算去畸变图像的边界
 * 
 * @param[in] imLeft            需要计算边界的图像
 */
void Frame::ComputeImageBounds(const cv::Mat &imLeft)	
{
    // 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    if(mDistCoef.at<float>(0)!=0.0)
    {
        // 保存矫正前的图像四个边界点坐标： (0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0;         //左上
        mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; //右上
        mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0;         //左下
        mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; //右下
        mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        // 和前面校正特征点一样的操作，将这几个边界点作为输入进行校正
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        //校正后的四个边界点已经不能够围成一个严格的矩形，因此在这个四边形的外侧加边框作为坐标的边界
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));//左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));//右上和右下横坐标最大的
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));//左上和右上纵坐标最小的
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));//左下和右下纵坐标最小的
    }
    else
    {
        // 如果畸变参数为0，就直接获得图像边界
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}


//当某个特征点的深度信息或者双目信息有效时,将它反投影到三维世界坐标系中
cv::Mat Frame::UnprojectStereo(const int &i)
{
    // KeyFrame::UnprojectStereo 
    // 貌似这里普通帧的反投影函数操作过程和关键帧的反投影函数操作过程有一些不同：
    // mvDepth是在ComputeStereoMatches函数中求取的
    // TODO 验证下面的这些内容. 虽然现在我感觉是理解错了,但是不确定;如果确定是真的理解错了,那么就删除下面的内容
    // mvDepth对应的校正前的特征点，可这里却是对校正后特征点反投影
    // KeyFrame::UnprojectStereo中是对校正前的特征点mvKeys反投影
    // 在ComputeStereoMatches函数中应该对校正后的特征点求深度？？ (wubo???)
    // NOTE 不过我记得好像上面的ComputeStereoMatches函数就是对于双目相机设计的，而双目相机的图像默认都是经过了校正的啊

    /** 步骤如下: <ul> */

    /** <li> 获取这个特征点的深度（这里的深度可能是通过双目视差得出的，也可能是直接通过深度图像的出来的） </li> */
    const float z = mvDepth[i];
    /** <li> 判断这个深度是否合法 </li> <ul> */
    //（其实这里也可以不再进行判断，因为在计算或者生成这个深度的时候都是经过检查了的_不行,RGBD的不是）
    if(z>0)
    {
        /** <li> 如果合法,就利用<b></b>矫正后的特征点的坐标 Frame::mvKeysUn 和相机的内参数,通过反投影和位姿变换得到空间点的坐标 </li> */
        //获取像素坐标，注意这里是矫正后的特征点的坐标
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        //计算在当前相机坐标系下的坐标
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        //生成三维点（在当前相机坐标系下）
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        //然后计算这个点在世界坐标系下的坐标，这里是对的，但是公式还是要斟酌一下。首先变换成在没有旋转的相机坐标系下，最后考虑相机坐标系相对于世界坐标系的平移
        return mRwc*x3Dc+mOw;
    }
    else
        /** <li> 如果深度值不合法，那么就返回一个空矩阵,表示计算失败 </li> */
        return cv::Mat();

}

} //namespace ORB_SLAM
