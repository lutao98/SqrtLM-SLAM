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
 * @file Frame.h
 * @author guoqing (1337841346@qq.com)
 * @brief ORB-SLAM2 中，帧的实现
 * @version 0.1
 * @date 2019-01-03 
 */

#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "data_structure/ORBVocabulary.h"
#include "FrontEnd/ORBextractor.h"
#include "data_structure/KeyFrame.h"
#include "data_structure/MapPoint.h"
#include "data_structure/point_types.h"
#include "utils/math_utils.h"
#include "utils/lidarconfig.h"

namespace ORB_SLAM2
{
struct ImageElement {
  int point_state = -3;   // -3=没有占用  -1=地面点  -2=聚类不稳定的类  0=非地面点
  int feature_state = 0;   // 1=角点/平面点
  int index = 0;
};

struct ImageIndex{
  int row = 0;
  int col = 0;
};
typedef std::pair<size_t, size_t> IndexRange;


/**
 * @name 定义一帧中有多少个图像网格
 */

/**
 * @brief 网格的行数
 */
#define FRAME_GRID_ROWS 48

/**
 * @brief 网格的列数
 */
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

/**
 * @brief 帧
 */
class Frame
{
public:

    /**
     * @brief Construct a new Frame object without parameter. 
     */
    Frame();

    // Copy constructor. 拷贝构造函数
    /**
     * @brief 拷贝构造函数
     * @details 复制构造函数, mLastFrame = Frame(mCurrentFrame)
     * 如果不是自定以拷贝函数的话，系统自动生成的拷贝函数对于所有涉及分配内存的操作都将是浅拷贝!!!
     * @param[in] frame 引用
     * @note 另外注意，调用这个函数的时候，这个函数中隐藏的this指针其实是指向目标帧的
     */
    Frame(const Frame &frame);

    /**
     * @brief 为单目相机准备的帧构造函数
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
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor,
          ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);


    // Constructor for Fusion(mono camera and lidar).
    /**
     * @brief 为Fusion准备的帧构造函数
     *
     * @param[in] imGray        对RGB图像灰度化之后得到的灰度图像
     * @param[in] imDepth       lidar深度图像
     * @param[in] timeStamp     时间戳
     * @param[in] extractor     特征点提取器句柄
     * @param[in] voc           ORB特征点词典的句柄
     * @param[in] K             相机的内参数矩阵
     * @param[in] distCoef      相机的去畸变参数
     * @param[in] bf            baseline*bf
     * @param[in] thDepth       远点和近点的深度区分阈值
     */
    Frame(const cv::Mat &imGray,
          pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_inputPtr,
          const lidarConfig* lidarconfig,
          const double &timeStamp,
          ORBextractor* extractor,
          ORBVocabulary* voc,
          cv::Mat &K,
          cv::Mat &distCoef,
          const float &bf,
          const float &thDepth,
          const Eigen::Matrix<double,3,4> &intrinsicMatrix,
          const Eigen::Matrix<double,4,4> &extrinsicMatrix);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    /**
     * @brief 提取图像的ORB特征，提取的关键点存放在mvKeys，描述子存放在mDescriptors
     * 
     * @param[in] flag          标记是左图还是右图。0：左图  1：右图
     * @param[in] im            等待提取特征点的图像
     */
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.
    // 存放在mBowVec中
    /**
     * @brief 计算词袋模型 
     * @details 计算词包 mBowVec 和 mFeatVec ，其中 mFeatVec 记录了属于第i个node（在第4层）的ni个描述子
     * @see CreateInitialMapMonocular() TrackReferenceKeyFrame() Relocalization()
     */
    void ComputeBoW();

    // Set the camera pose.
    // 用Tcw更新mTcw
    /**
     * @brief 用 Tcw 更新 mTcw 以及类中存储的一系列位姿
     * 
     * @param[in] Tcw 从世界坐标系到当前帧相机位姿的变换矩阵
     */
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    /**
     * @brief 根据相机位姿,计算相机的旋转,平移和相机中心等矩阵.
     * @details 其实就是根据Tcw计算mRcw、mtcw和mRwc、mOw.
     */
    void UpdatePoseMatrices();

    // Returns the camera center.
    /**
     * @brief 返回位于当前帧位姿时,相机的中心
     * 
     * @return cv::Mat 相机中心在世界坐标系下的3D点坐标
     */
    inline cv::Mat GetCameraCenter()
    {
        return mOw.clone();
    }

    /**
     * @brief Get the Rotation Inverse object
     * mRwc存储的是从当前相机坐标系到世界坐标系所进行的旋转，而我们一般用的旋转则说的是从世界坐标系到当前相机坐标系的旋转
     * @return 返回从当前帧坐标系到世界坐标系的旋转
     */
    inline cv::Mat GetRotationInverse()
    {
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    /**
     * @brief 判断路标点是否在视野中
     * 步骤
     * Step 1 获得这个地图点的世界坐标
     * Step 2 关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，返回false
     * Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
     * Step 4 关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
     * Step 5 关卡四：计算当前视角和“法线”夹角的余弦值, 若小于设定阈值，返回false
     * Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
     * Step 7 记录计算得到的一些参数
     * @param[in] pMP                       当前地图点
     * @param[in] viewingCosLimit           夹角余弦，用于限制地图点和光心连线和法线的夹角
     * @return true                         地图点合格，且在视野内
     * @return false                        地图点不合格，抛弃
     */
    bool isInFrustum(MapPoint *pMP, float viewingCosLimit);

    /**
     * @brief 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
     * 
     * @param[in] kp                    给定的特征点
     * @param[in & out] posX            特征点所在网格坐标的横坐标
     * @param[in & out] posY            特征点所在网格坐标的纵坐标
     * @return true                     如果找到特征点所在的网格坐标，返回true
     * @return false                    没找到返回false
     */
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    /**
     * @brief 找到在 以x,y为中心,半径为r的圆形内且金字塔层级在[minLevel, maxLevel]的特征点
     * 
     * @param[in] x                     特征点坐标x
     * @param[in] y                     特征点坐标y
     * @param[in] r                     搜索半径 
     * @param[in] minLevel              最小金字塔层级
     * @param[Framein] maxLevel              最大金字塔层级
     * @return vector<size_t>           返回搜索到的候选匹配点id
     */
    vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    /**
     * @brief 当某个特征点的深度信息或者双目信息有效时,将它反投影到三维世界坐标系中
     * 
     * @param[in] i     特征点的ID
     * @return cv::Mat  反投影后得到的特征点的反投影点的世界坐标
     */
    cv::Mat UnprojectStereo(const int &i);

    cv::Mat getDepthimg();
    cv::Mat getRangeImage();
    PointIRTCloud getCornerPoints();
    PointIRTCloud getFlatPoints();
    PointIRTCloud getLessCornerPoints();
    PointIRTCloud getLessFlatPoints();
public:
    const lidarConfig* lidarconfig_;
    cv::Mat Depthimg_;
    cv::Mat range_img_visual_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_inputPtr_;
    vector<PointIRTCloudPtr> all_laser_scans_;
    vector<PointIRTCloudPtr> laser_scans_;
    // 每行范围
    vector<IndexRange> scan_ranges_;
    vector<vector<ImageElement>> range_image_;
    vector<vector<ImageIndex>> image_index_;
    vector<ImageIndex> surface_points_less_flat_index_;
    vector<vector<float>> ringAng;
    vector<float> scanAng;
    PointIRTCloud cloud_in_rings_;
    PointIRTCloud corner_points_sharp_;
    PointIRTCloud corner_points_less_sharp_;
    PointIRTCloud surface_points_flat_;
    PointIRTCloud surface_points_less_flat_;
    PointIRTCloud surface_points_flat_normal_;
    PointIRTCloud surface_points_less_flat_normal_;
    vector<int> scan_ring_mask_;
    vector<pair<float, size_t> > curvature_idx_pairs_; // in subregion
    vector<vector<ImageIndex>> less_sharp_;
    vector<ImageIndex> sharp_;

    // Vocabulary used for relocalization.
    ///用于重定位的ORB特征字典
    ORBVocabulary *mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ///ORB特征提取器句柄,其中右侧的提取器句柄只会在双目输入的情况中才会被用到
    ORBextractor *mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    ///帧的时间戳
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.

    /**
     * @name 相机的内参数
     */

    ///相机的内参数矩阵
    cv::Mat mK;
    //NOTICE 注意这里的相机内参数其实都是类的静态成员变量；此外相机的内参数矩阵和矫正参数矩阵却是普通的成员变量，
    //NOTE 这样是否有些浪费内存空间？

    static float fx;        ///<x轴方向焦距
    static float fy;        ///<y轴方向焦距
    static float cx;        ///<x轴方向光心偏移
    static float cy;        ///<y轴方向光心偏移
    static float invfx;     ///<x轴方向焦距的逆
    static float invfy;     ///<x轴方向焦距的逆

    //TODO 目测是opencv提供的图像去畸变参数矩阵的，但是其具体组成未知
    ///去畸变参数
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    ///baseline x fx
    float mbf;

    // Stereo baseline in meters.
    ///相机的基线长度,单位为米
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    //TODO 这里它所说的话还不是很理解。尤其是后面的一句。
    //而且,这个阈值不应该是在哪个帧中都一样吗?
    ///判断远点和近点的深度阈值
    float mThDepth;

    // Number of KeyPoints.
    int N; 

    /**
     * @name 关于特征点
     */

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    // mvKeys:原始左图像提取出的特征点（未校正）
    // mvKeysRight:原始右图像提取出的特征点（未校正）
    // mvKeysUn:校正mvKeys后的特征点，对于双目摄像头，一般得到的图像都是校正好的，再校正一次有点多余

    ///原始左图像提取出的特征点（未校正）
    std::vector<cv::KeyPoint> mvKeys;
    ///原始右图像提取出的特征点（未校正）
    std::vector<cv::KeyPoint> mvKeysRight;
    ///校正mvKeys后的特征点
    std::vector<cv::KeyPoint> mvKeysUn;

    ///@note 之所以对于双目摄像头只保存左图像矫正后的特征点,是因为对于双目摄像头,一般得到的图像都是矫正好的,这里再矫正一次有些多余.\n
    ///校正操作是在帧的构造函数中进行的。
    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    // 对于双目，mvuRight存储了左目像素点在右目中的对应点的横坐标 （因为纵坐标是一样的）
    // mvDepth对应的深度
    // 单目摄像头，这两个容器中存的都是-1

    ///@note 对于单目摄像头，这两个容器中存的都是-1
    ///对于双目相机,存储左目像素点在右目中的对应点的横坐标 （因为纵坐标是一样的）

    std::vector<float> mvuRight;	//m-member v-vector u-指代横坐标,因为最后这个坐标是通过各种拟合方法逼近出来的，所以使用float存储
    ///对应的深度
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    // 内部实际存储的是std::map<WordId, WordValue>
    // WordId 和 WordValue 表示Word在叶子中的id 和权重
    DBoW2::BowVector mBowVec;
    // 内部实际存储 std::map<NodeId, std::vector<unsigned int> >
    // NodeId 表示节点id，std::vector<unsigned int> 中实际存的是该节点id下所有特征点在图像中的索引
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    /// 左目摄像头和右目摄像头特征点对应的描述子
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    /// 每个特征点对应的MapPoint.如果特征点没有对应的地图点,那么将存储一个空指针
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    // 观测不到Map中的3D点
    /// 属于外点的特征点标记,在 Optimizer::PoseOptimization 使用了
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    //原来通过对图像分区域还能够降低重投影地图点时候的匹配复杂度啊。。。。。
    ///@note 注意到上面也是类的静态成员变量， 有一个专用的标志mbInitialComputations用来在帧的构造函数中标记这些静态成员变量是否需要被赋值
    /// 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementWidthInv;
    /// 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementHeightInv;

    // 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
    ///这个向量中存储的是每个图像网格内特征点的id（左图）
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw; ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵,是我们常规理解中的相机位姿

    // Current and Next Frame id.
    // 类的静态成员变量，这些变量则是在整个系统开始执行的时候被初始化的——它在全局区被初始化
    static long unsigned int nNextId; ///< Next Frame id.
    long unsigned int mnId; ///< Current Frame id.

    // Reference Keyframe.
    // 普通帧与自己共视程度最高的关键帧作为参考关键帧
    KeyFrame* mpReferenceKF;

    /**
     * @name 图像金字塔信息
     * @{
     */
    // Scale pyramid info.
    int mnScaleLevels;                  ///<图像金字塔的层数
    float mfScaleFactor;                ///<图像金字塔的尺度因子
    float mfLogScaleFactor;             ///<图像金字塔的尺度因子的对数值，用于仿照特征点尺度预测地图点的尺度

    std::vector<float> mvScaleFactors;		///<图像金字塔每一层的缩放因子
    std::vector<float> mvInvScaleFactors;	///<以及上面的这个变量的倒数
    std::vector<float> mvLevelSigma2;		///@todo 目前在frame.c中没有用到，无法下定论
    std::vector<float> mvInvLevelSigma2;		///<上面变量的倒数

    // Undistorted Image Bounds (computed once).
    /**
     * @name 用于确定画格子时的边界 
     * @note（未校正图像的边界，只需要计算一次，因为是类的静态成员变量）
     * @{
     */
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    /**
     * @brief 一个标志，标记是否已经进行了这些初始化计算
     * @note 由于第一帧以及SLAM系统进行重新校正后的第一帧会有一些特殊的初始化处理操作，所以这里设置了这个变量. \n
     * 如果这个标志被置位，说明再下一帧的帧构造函数中要进行这个“特殊的初始化操作”，如果没有被置位则不用。
    */
    static bool mbInitialComputations;

private:

    void imageProcess();
    void lidarProcess();
    void CalculateRingAndTime(const PointICloud &all_cloud_in, PointIRTCloud &all_cloud_out);
    void PointToImage(const PointIRTCloud &all_cloud_in);
    void PrepareRing_corner(const PointIRTCloud &scan);
    void PrepareRing_flat(const PointIRTCloud &scan);
    void PrepareSubregion_corner(const PointIRTCloud &scan, const size_t idx_start, const size_t idx_end);
    void PrepareSubregion_flat(const PointIRTCloud &scan, const size_t idx_start, const size_t idx_end);
    void ExtractFeaturePoints();

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    /**
     * @brief 用内参对特征点去畸变，结果报存在mvKeysUn中
     * 
     */
    void UndistortKeyPoints();

    /**
     * @brief 计算去畸变图像的边界
     * 
     * @param[in] imLeft            需要计算边界的图像
     */
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    /**
     * @brief 将提取到的特征点分配到图像网格中 \n
     * @details 该函数由构造函数调用
     * 
     */
    void AssignFeaturesToGrid();

    /**
     * @name 和相机位姿有关的变量
     * @{
     */
    // Rotation, translation and camera center
    cv::Mat mRcw; ///< Rotation from world to camera
    cv::Mat mtcw; ///< Translation from world to camera
    cv::Mat mRwc; ///< Rotation from camera to world
    cv::Mat mOw;  ///< mtwc,Translation from camera to world

    /** @} */
};

}// namespace ORB_SLAM



#endif // FRAME_H
