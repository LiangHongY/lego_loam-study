// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.
#include"utility.h"

#include<gtsam/geometry/Rot3.h>
#include<gtsam/geometry/Pose3.h>
#include<gtsam/slam/PriorFactor.h>
#include<gtsam/slam/BetweenFactor.h>
#include<gtsam/nonlinear/NonlinearFactorGraph.h>
#include<gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include<gtsam/nonlinear/Marginals.h>
#include<gtsam/nonlinear/Values.h>

#include<gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

class mapOptimization{

    private:
        
        NonlinearFactorGraph gtSAMgraph;
        Values initialEstimate;
        Values optimizedEstimate;
        ISAM2 *isam;
        Values isamCurrentEstimate;

        noiseModel::Diagonal::shared_ptr priorNoise;
        noiseModel::Diagonal::shared_ptr odometryNoise;
        noiseModel::Diagonal::shared_ptr constraintNoise;

        ros::NodeHandle nh;

        ros::Publisher pubLaserCloudSurround;
        ros::Publisher pubOdomAftMapped;
        ros::Publisher pubKeyPoses;

        ros::Publisher pubHistoryKeyFrames;
        ros::Publisher pubIcpKeyFrames;
        ros::Publisher pubRecentKeyFrames;
        ros::Publisher pubRegisteredCloud;

        ros::Subscriber subLaserCloudCornerLast;
        ros::Subscriber subLaserCloudSurfLast;
        ros::Subscriber subOutlierCloudLast;
        ros::Subscriber subLaserOdometry;
        ros::Subscriber subImu;

        nav_msgs::Odometry odomAftMapped;
        tf::StampedTransform aftMappedTrans;
        tf::TransformBroadcaster tfBroadcaster;

        vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
        vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
        vector<pcl::PointCloud<PointType>::Ptr> outlierCloudKeyFrames;

        deque<pcl::PointCloud<PointType>::Ptr> recentCornerCloudKeyFrames;
        deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;
        deque<pcl::PointCloud<PointType>::Ptr> recentOutlierCloudKeyFrames;
        int latestFrameID;

        vector<int> surroundingExistingKeyPosesID;
        deque<pcl::PointCloud<PointType>::Ptr> surroundingCornerCloudKeyFrames;
        deque<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyFrames;
        deque<pcl::PointCloud<PointType>::Ptr> surroundingOutlierCloudKeyFrames;

        PointType previousRobotPosPoint;
        PointType currentRobotPosPoint;

        pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;//220306can PointType的XYZI分别保存３个方向上的平移和一个索引(cloudKeyPoses3D->points.size())
        pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;//220306can　PointTypePose的XYZI保存和cloudKeyPoses3D一样的内容，另外还保存RPY角度以及一个时间值timeLaserOdometry


        //220306can　结尾有DS代表是downsimple,下采样
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;

        pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;   // corner feature set from odoOptimization
        pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;     // surf feature set from odoOptimization
        pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
        pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;   //(downsampled) surf feature set from odoOptimization
        
        pcl::PointCloud<PointType>::Ptr laserCloudOutlierLast;      //corner feature set from odoOptimization
        pcl::PointCloud<PointType>::Ptr laserCloudOutlierLastDS;    //downsampled coner feature set from odoOptimization

        pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLast;    //surf feature set from odoOptimization
        pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS;  //downsampled surf feature set from odoOptimization

        pcl::PointCloud<PointType>::Ptr laserCloudOri;
        pcl::PointCloud<PointType>::Ptr coeffSel;

        pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
        pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
        pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
        pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;


        pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloud;
        pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloudDS;
        pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloud;
        pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloudDS;

        pcl::PointCloud<PointType>::Ptr latestCornerKeyFrameCloud;
        pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloud;
        pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloudDS;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS;
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        pcl::VoxelGrid<PointType> downSizeFilterCorner;
        pcl::VoxelGrid<PointType> downSizeFilterSurf;
        pcl::VoxelGrid<PointType> downSizeFilterOutlier;
        pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames;   //for history key frames of loop closure
        pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;    //for surrounding key poses of scan-to-map optimization
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;      // for global map visualization
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;     //for global map visualization

        double timeLaserCloudCornerLast;
        double timeLaserCloudSurfLast;
        double timeLaserOdometry;
        double timeLaserCloudOutlierLast;
        double timeLaserGloalMapPublish;

        bool newLaserCloudCornerLast;
        bool newLaserCloudSurfLast;
        bool newLaserOdometry;
        bool newLaserCloudOutlierLast;


        float transformLast[6];
        /*************高频转换量**************/
        //220306can  odometry计算得到的到世界坐标系下的转移矩阵
        float transformSum[6];
        //220306can  转移增量，只使用了后三个平移增量
        float transformIncre[6];
        /*************低频转换量*************/
        //220306can 以起始位置为原点的世界坐标系下的转换矩阵（猜测与调整的对象?这里个人没有理解）
        float transformTobeMapped[6];
        //220306can 存放mapping之前的Odometry计算的世界坐标系的转换矩阵（注：低频量，不一定与transformSum一样）
        float transformBefMapped[6];
        //220306can 存放mapping之后的经过mapping微调之后的转换矩阵
        float transformAftMapped[6];        //220226这几个装的是啥？可以参考上面的解析


        int imuPointerFront;
        int imuPointerLast;

        double imuTime[imuQueLength];
        float imuRoll[imuQueLength];
        float imuPitch[imuQueLength];

        std::mutex mtx;

        double timeLastProcessing;

        PointType pointOri, pointSel, pointProj, coeff;

        cv::Mat matA0;  // 表示的是高斯牛顿法里面的迭代计算使用到的雅克比矩阵
        cv::Mat matB0;
        cv::Mat matX0;

        cv::Mat matA1;
        cv::Mat matD1;
        cv::Mat matV1;

        bool isDegenerate;
        cv::Mat matP;

        int laserCloudCornerFromMapDSNum;
        int laserCloudSurfFromMapDSNum;
        int laserCloudCornerLastDSNum;
        int laserCloudSurfLastDSNum;
        int laserCloudOutlierLastDSNum;
        int laserCloudSurfTotalLastDSNum;

        bool potentialLoopFlag;
        double timeSaveFirstCurrentScanForLoopClosure;
        int closestHistoryFrameID;
        int latestFrameIDLoopCloure;

        bool aLoopIsClosed;

        float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;
        float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;
    
    public:


        //220305  初始化
        mapOptimization():
            nh("~")
        {   //220306can 用于闭环图优化的参数设置，使用gtsam库
            ISAM2Params parameters;
                parameters.relinearizeThreshold = 0.01;
                parameters.relinearizeSkip = 1;
            isam = new ISAM2(parameters);

            pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
            pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
            pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 5);

            //2220405can  lidar建图接收５个话题，包括特征提取之后的点云，lidar里程计和IMU消息
            subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, &mapOptimization::laserCloudCornerLastHandler, this);
            subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, &mapOptimization::laserCloudSurfLastHandler, this);
            subOutlierCloudLast = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2,&mapOptimization::laserCloudOutlierLastHandler, this);
            subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laserOdometryHandler, this);
            subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 50, &mapOptimization::imuHandler, this);

            pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/history_cloud",2);
            pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
            pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);
            pubRegisteredCloud = nh.advertise<sensor_msgs::PointCloud2>("/registered_cloud", 2);
            //220306can  设置滤波时创建的体素大小为0.2m/0.4m立方体,下面的单位为m
            downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);    //角下采样
            downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);      //面下采样，220227这些采样参数啥意思？
            downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);   //外部点下采样

            downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4);  //for history key frames of loop closure
            downSizeFilterSurroundingKeyPoses.setLeafSize(1.0, 1.0, 1.0);   //for surrounding key poses of scan-to-map optimization

            downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0);     //for global map visualization
            downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4);    //for global map visualization

            odomAftMapped.header.frame_id = "/camera_init";
            odomAftMapped.child_frame_id = "/aft_mapped";

            aftMappedTrans.frame_id_ = "/camera_init";
            aftMappedTrans.child_frame_id_ = "/aft_mapped";

            allocateMemory();   //分配内存
        }

        void allocateMemory(){   //220305 分配内存
            
            cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
            cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

            kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
            kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

            surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
            surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());

            laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());   //corner feature set from odoOptimization
            laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());     //surf feature set from odoOptimization
            laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner frature set from odoOptimization
            laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());   //downsmapled surf feature set from odoOptimization
            laserCloudOutlierLast.reset(new pcl::PointCloud<PointType>());
            laserCloudOutlierLastDS.reset(new pcl::PointCloud<PointType>());
            laserCloudSurfTotalLast.reset(new pcl::PointCloud<PointType>());
            laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>());

            laserCloudOri.reset(new pcl::PointCloud<PointType>());
            coeffSel.reset(new pcl::PointCloud<PointType>());

            laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
            laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
            laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
            laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

            kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
            kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());


            nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
            nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
            nearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
            nearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

            latestCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
            latestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
            latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

            kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
            globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
            globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
            globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
            globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

            timeLaserCloudCornerLast = 0;
            timeLaserCloudSurfLast = 0;
            timeLaserOdometry = 0;
            timeLaserCloudOutlierLast = 0;
            timeLaserGloalMapPublish = 0;

            timeLastProcessing = -1;

            newLaserCloudCornerLast = false;
            newLaserCloudSurfLast = false;

            newLaserOdometry = false;
            newLaserCloudOutlierLast = false;

            for(int i = 0; i < 6; ++i){
                transformLast[i] = 0;
                transformSum[i] = 0;
                transformIncre[i] = 0;
                transformTobeMapped[i] = 0;
                transformBefMapped[i] = 0;
                transformAftMapped[i] = 0;
            }

            imuPointerFront = 0;
            imuPointerLast = -1;    //220227 这两个关于imu是含义参数呢？

            for(int i = 0; i < imuQueLength; ++i){
                imuTime[i] = 0;
                imuRoll[i] = 0;
                imuPitch[i] = 0;       
            }

            gtsam::Vector Vector6(6);
            Vector6 << 1e-6, 1e-6, 1e-6, 1e-8,1e-8,1e-6;    //220227 初始化？跟标准的vector有啥区别？是Eigen库里面的，其和标准的vector含义一样，只不过多了一个定义长度的
            priorNoise = noiseModel::Diagonal::Variances(Vector6);  //220227 定义噪声？？？
            odometryNoise = noiseModel::Diagonal::Variances(Vector6);

            matA0 = cv::Mat (5, 3, CV_32F, cv::Scalar::all(0));     //220227 括号内的参数含义是啥？
            matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
            matX0 = cv::Mat (3, 1, CV_32F, cv::Scalar::all(0));

            matA1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0)); //220306can matA1为边缘特征的协方差矩阵
            matD1 = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0)); //220306can matA1的特征值
            matV1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0)); //220306can matA1的特征向量，对应于matD1存储(个人后半句不理解)

            isDegenerate = false;
            matP = cv::Mat (6, 6, CV_32F, cv::Scalar::all(0));

            laserCloudCornerFromMapDSNum = 0;
            laserCloudSurfFromMapDSNum = 0;
            laserCloudCornerLastDSNum = 0;
            laserCloudSurfLastDSNum = 0;
            laserCloudOutlierLastDSNum = 0;
            laserCloudSurfTotalLastDSNum = 0;

            potentialLoopFlag = false;
            aLoopIsClosed = false;

            latestFrameID = 0;
        }
        //220306can 将坐标转移到世界坐标系下,得到可用于建图的Lidar坐标，即修改了transformTobeMapped的值
        void transformAssociateToMap()      // 220305 将相关特征的转化到地图中？？？ 解析上上一行
        {
            float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3])
                        - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);//220227　计算的是啥量值？
            float y1 = transformBefMapped[4] - transformSum[4];
            float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3])
                        + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
            
            float x2 = x1;
            float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
            float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

            transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
            transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
            transformIncre[5] = z2; //220227 这些变量含义是啥？

            float sbcx = sin(transformSum[0]);
            float cbcx = cos(transformSum[0]);
            float sbcy = sin(transformSum[1]);
            float cbcy = cos(transformSum[1]);
            float sbcz = sin(transformSum[2]);
            float cbcz = cos(transformSum[2]);

            float sblx = sin(transformBefMapped[0]);
            float cblx = cos(transformBefMapped[0]);
            float sbly = sin(transformBefMapped[1]);
            float cbly = cos(transformBefMapped[1]);
            float sblz = sin(transformBefMapped[2]);
            float cblz = cos(transformBefMapped[2]);

            float salx = sin(transformAftMapped[0]);
            float calx = sin(transformAftMapped[0]);
            float saly = sin(transformAftMapped[1]);
            float caly = sin(transformAftMapped[1]);
            float salz = sin(transformAftMapped[2]);
            float calz = sin(transformAftMapped[2]);

            float srx = -sbcx*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz)
                        - cbcx*sbcy*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                        - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                        - cbcx*cbcy*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                        - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx);
            transformTobeMapped[0] = -asin(srx);
            //220227 下面公式直接copy过来的，具体含义压根没懂
            float srycrx = sbcx*(cblx*cblz*(caly*salz - calz*salx*saly)
                        - cblx*sblz*(caly*calz + salx*saly*salz) + calx*saly*sblx)
                        - cbcx*cbcy*((caly*calz + salx*saly*salz)*(cblz*sbly - cbly*sblx*sblz)
                        + (caly*salz - calz*salx*saly)*(sbly*sblz + cbly*cblz*sblx) - calx*cblx*cbly*saly)
                        + cbcx*sbcy*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
                        + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) + calx*cblx*saly*sbly);
            float crycrx = sbcx*(cblx*sblz*(calz*saly - caly*salx*salz)
                        - cblx*cblz*(saly*salz + caly*calz*salx) + calx*caly*sblx)
                        + cbcx*cbcy*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
                        + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) + calx*caly*cblx*cbly)
                        - cbcx*sbcy*((saly*salz + caly*calz*salx)*(cbly*sblz - cblz*sblx*sbly)
                        + (calz*saly - caly*salx*salz)*(cbly*cblz + sblx*sbly*sblz) - calx*caly*cblx*sbly);
            transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]), 
                                        crycrx / cos(transformTobeMapped[0]));
            
            float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                        - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                        - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                        - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                        + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
            float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                        - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                        - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                        - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                        + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
            transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]), 
                                        crzcrx / cos(transformTobeMapped[0]));

            x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
            y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
            z1 = transformIncre[5];

            x2 = x1;
            y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
            z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

            transformTobeMapped[3] = transformAftMapped[3] 
                                - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
            transformTobeMapped[4] = transformAftMapped[4] - y2;
            transformTobeMapped[5] = transformAftMapped[5] 
                                - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
        }
    
        void transformUpdate()//220227 转换时间更新？？？
        {
            if(imuPointerLast >= 0){
                float imuRollLast = 0,imuPitchLast = 0;
                while (imuPointerFront != imuPointerLast){
                    if(timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]){
                        break;
                    }
                    imuPointerFront = (imuPointerFront + 1) % imuQueLength;
                }
                //220227 这些判断时间目地为了啥？
                if(timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]){
                    imuRollLast = imuRoll[imuPointerFront];
                    imuPitchLast = imuPitch[imuPointerFront];
                }else{
                    int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                    float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack])
                                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                    float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod)
                                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                    
                    imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
                    imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
                }

                transformTobeMapped[0] = 0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
                transformTobeMapped[2] = 0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
            }

            for(int i = 0; i < 6; i++){
                transformBefMapped[i] = transformSum[i];        //220227 地图转换前时间？？？
                transformAftMapped[i] = transformTobeMapped[i]; //220227  地图转换后时间？？？
            }
        }
        //220306can 先提前求好roll,pitch,yaw的sin和cos值
        void updatePointAssociateToMapSinCos(){//220227 更新点转换地图相关sin,cos值？？？
            cRoll = cos(transformTobeMapped[0]);
            sRoll = sin(transformTobeMapped[0]);

            cPitch = cos(transformTobeMapped[1]);
            sPitch = sin(transformTobeMapped[1]);

            cYaw = cos(transformTobeMapped[2]);
            sYaw = sin(transformTobeMapped[2]);

            tX = transformTobeMapped[3];
            tY = transformTobeMapped[4];
            tZ = transformTobeMapped[5];
        }

        void pointAssociateToMap(PointType const * const pi,PointType * const po)   //220227 转换到地图时，点相关信息的更新？？？　局部点转全局点计算
        {
            //220306can map可以意味着全局
            // 进行6自由度的变换，先进行旋转（顺序Ｚ－＞Ｘ－＞Ｙ），然后再平移
            // 主要进行坐标变换，将局部坐标转换到全局坐标中去	

            // 先绕z轴旋转
            //     |cosrz  -sinrz  0|
            //  Rz=|sinrz  cosrz   0|
            //     |0       0      1|
            // [x1,y1,z1]^T=Rz*[pi->x,pi->y,pi->z]
            // 220306can   
            float x1 = cYaw * pi->x - sYaw * pi->y;
            float y1 = sYaw * pi->x - cYaw * pi->y;
            float z1 = pi->z;

            //220306can
            //绕Ｘ轴旋转
            // [x2,y2,z2]^T=Rx*[x1,y1,z1]
            //    |1     0        0|
            // Rx=|0   cosrx -sinrx|
            //    |0   sinrx  cosrx|
            float x2 = x1;
            float y2 = cRoll * y1 - sRoll * z1;
            float z2 = sRoll * y1 + cRoll * z1;

            //220306can
            // 最后再绕Y轴旋转，然后加上平移
            //    |cosry   0   sinry|
            // Ry=|0       1       0|
            //    |-sinry  0   cosry|
            po->x = cPitch * x2 + sPitch * z2 + tX;
            po->y = y2 + tY;
            po->z = -sPitch * x2 +cPitch * z2 + tZ;
            po->intensity = pi->intensity; 
        }

        void updateTransformPointCloudSinCos(PointTypePose *tIn){//220227 更新转换到地图时，点的sin,cos角度变换？？？　全局坐标系下的sin,cos值更新

            ctRoll = cos(tIn->roll);
            stRoll = sin(tIn->roll);

            ctPitch = cos(tIn->pitch);
            stPitch = sin(tIn->pitch);

            ctYaw = cos(tIn->yaw);
            stYaw = sin(tIn->yaw);

            tInX = tIn->x;
            tInY = tIn->y;
            tInZ = tIn->z;
        }

        pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn){
            //!!! Do not use pcl for point cloud transformation,results are not accurate
            //reason: unknow
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

            PointType *pointFrom;
            PointType pointTo;

            int cloudSize = cloudIn->points.size();
            cloudOut->resize(cloudSize);

            for(int i = 0; i < cloudSize; ++i){

                pointFrom = &cloudIn->points[i];
                float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
                float y1 = stYaw * pointFrom->x + ctYaw * pointFrom->y;
                float z1 = pointFrom->z;

                float x2 = x1;
                float y2 = ctRoll * y1 - stRoll * z1;
                float z2 = stRoll * y1 + ctRoll * z1;
                //220228 下面几个变量如何理解？？？是指一个点从原始坐标系转到另一个坐标系下的表示？？？
                pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
                pointTo.y = y2 + tInY;
                pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
                pointTo.intensity = pointFrom->intensity;

                cloudOut->points[i] = pointTo;
            }
            return cloudOut;
        }
        //220228 重载函数，但是作用是啥？？？？判断输入变量的个数来决定转变类型？？？？
        pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,PointTypePose* transformIn){

            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

            PointType *pointFrom;
            PointType pointTo;

            int cloudSize = cloudIn->points.size();
            cloudOut->resize(cloudSize);
            //220306can 坐标系变换，旋转rpy角
            for(int i = 0; i < cloudSize; ++i){

                pointFrom = &cloudIn->points[i];
                float x1 = cos(transformIn->yaw) * pointFrom->x - sin(transformIn->yaw) * pointFrom->y;
                float y1 = sin(transformIn->yaw) * pointFrom->x + cos(transformIn->yaw) * pointFrom->y;
                float z1 = pointFrom->z;

                float x2 = x1;
                float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
                float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll) * z1;

                pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 + transformIn->x;
                pointTo.y = y2 + transformIn->y;
                pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) *z2 +transformIn->z;
                pointTo.intensity = pointFrom->intensity;

                cloudOut->points[i] = pointTo;
            }
            return cloudOut;
        }

        void laserCloudOutlierLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){     //220305    外部点handler获取
            timeLaserCloudCornerLast = msg->header.stamp.toSec();
            laserCloudOutlierLast->clear();
            pcl::fromROSMsg(*msg, *laserCloudOutlierLast);  //220228 将ros类型的点云信息格式转为pcl类型
            newLaserCloudOutlierLast = true;
        }

        void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){      //220305    角点handler获取
            timeLaserCloudCornerLast = msg->header.stamp.toSec();
            laserCloudCornerLast->clear();
            pcl::fromROSMsg(*msg, *laserCloudCornerLast);
            newLaserCloudCornerLast = true;
        }

        void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){        //220305    面点handler获取
            timeLaserCloudSurfLast = msg->header.stamp.toSec();
            laserCloudSurfLast->clear();
            pcl::fromROSMsg(*msg, *laserCloudSurfLast);
            newLaserCloudSurfLast = true;
        }

        void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry){       //220305    雷达odom的handler获取
            timeLaserOdometry = laserOdometry->header.stamp.toSec();
            double roll, pitch, yaw;
            geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;//220228 激光的odom位姿转四元数表示
            tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, geoQuat.y,geoQuat.w)).getRPY(roll, pitch, yaw);//220228 欧拉角转四元数表示
            transformSum[0] = -pitch;
            transformSum[1] = -yaw;
            transformSum[2] = roll;
            transformSum[3] = laserOdometry->pose.pose.position.x;
            transformSum[4] = laserOdometry->pose.pose.position.y;
            transformSum[5] = laserOdometry->pose.pose.position.z;
            newLaserOdometry = true;
        }

        void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn){                           //220305    imu的handler获取
            double roll, pitch, yaw;
            tf::Quaternion orientation;
            tf::quaternionMsgToTF(imuIn->orientation, orientation);//220228 四元数转tf
            tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
            imuPointerLast = (imuPointerLast + 1) % imuQueLength;
            imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
            imuRoll[imuPointerLast] = roll;
            imuPitch[imuPointerLast] = pitch;
        }
            
        void publishTF(){                       //220305 发布TF信息

            geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                        (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

            odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
            odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
            odomAftMapped.pose.pose.orientation.z = geoQuat.x;
            odomAftMapped.pose.pose.orientation.w = geoQuat.w;
            odomAftMapped.pose.pose.position.x = transformAftMapped[3];
            odomAftMapped.pose.pose.position.y = transformAftMapped[4];
            odomAftMapped.pose.pose.position.z = transformAftMapped[5];
            odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
            odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
            odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
            odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
            odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
            odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
            pubOdomAftMapped.publish(odomAftMapped);

            aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
            aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.w));
            aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
            tfBroadcaster.sendTransform(aftMappedTrans);
        }

        PointTypePose trans2PointTypePose(float transformIn[]){         //220305　转换点的位姿？？？
            PointTypePose thisPose6D;
            thisPose6D.x = transformIn[3];
            thisPose6D.y = transformIn[4];
            thisPose6D.z = transformIn[5];
            thisPose6D.roll = transformIn[0];
            thisPose6D.pitch = transformIn[1];
            thisPose6D.yaw = transformIn[2];
            return thisPose6D;
        }

        void publishKeyPosesAndFrames(){    //220301 发布关键位姿和帧

            if(pubKeyPoses.getNumSubscribers() != 0){
                sensor_msgs::PointCloud2 cloudMsgTemp;
                pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
                cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                cloudMsgTemp.header.frame_id = "/camera_init";
                pubKeyPoses.publish(cloudMsgTemp);
            }

            if(pubRecentKeyFrames.getNumSubscribers() != 0){
                sensor_msgs::PointCloud2 cloudMsgTemp;
                pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
                cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                cloudMsgTemp.header.frame_id = "/camera_init";
                pubRecentKeyFrames.publish(cloudMsgTemp);
            }

            if(pubRegisteredCloud.getNumSubscribers() != 0){
                pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
                *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
                *cloudOut += *transformPointCloud(laserCloudSurfTotalLast, &thisPose6D);

                sensor_msgs::PointCloud2 cloudMsgTemp;
                pcl::toROSMsg(*cloudOut, cloudMsgTemp);//220301 转为ros消息
                cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                cloudMsgTemp.header.frame_id = "/camera_init";
                pubRegisteredCloud.publish(cloudMsgTemp);
            }
        }
        
        void visualizeGlobalMapThread(){    //220301 可视化全局地图线程．把边＼面点云保存到pcd里面，这些点云经过了下采样的
            ros::Rate rate(0.2);
            while(ros::ok()){
                rate.sleep();
                publishGlobalMap();
            }
            //save final point cloud
            pcl::io::savePCDFileASCII(fileDirectory+"finalCloud.pcd", *globalMapKeyFramesDS);

            string cornerMapString = "/tmp/cornerMap.pcd";
            string surfaceMapString = "/tmp/surfaceMap.pcd";
            string trajectoryString = "/tmp/trajectory.pcd";

            pcl::PointCloud<PointType>::Ptr cornerMapCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr cornerMapCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surfaceMapCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surfaceMapCloudDS(new pcl::PointCloud<PointType>());

            for(int i = 0; i < cornerCloudKeyFrames.size(); i++){
                *cornerMapCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
                *surfaceMapCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
                *surfaceMapCloud += *transformPointCloud(outlierCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            }

            downSizeFilterCorner.setInputCloud(cornerMapCloud);
            downSizeFilterCorner.filter(*cornerMapCloudDS);
            downSizeFilterSurf.setInputCloud(surfaceMapCloud);
            downSizeFilterSurf.filter(*surfaceMapCloudDS);

            pcl::io::savePCDFileASCII(fileDirectory+"cornerMap.pcd", *cornerMapCloudDS);
            pcl::io::savePCDFileASCII(fileDirectory+"surfaceMap.pcd", *surfaceMapCloudDS);
            pcl::io::savePCDFileASCII(fileDirectory+"trajectory.pcd", *cloudKeyPoses3D);
        }
        
        void publishGlobalMap(){    //220301 发布全局地图

            if(pubLaserCloudSurround.getNumSubscribers() == 0)
                return;
            
            if(cloudKeyPoses3D->points.empty() == true)
                return;
                //kd-tree to find near key frames to visualize 220228 kd树寻找相邻帧以可视化
            std::vector<int> pointSearchIndGlobalMap;
            std::vector<float> pointSearchSqDisGlobalMap;
                //search near key frames to visualize
            mtx.lock();     //220301 锁住进程
            kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);//220306can 通过KDTree进行最近邻搜索
            kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
            mtx.unlock();

            for(int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
                globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
                    //downsample near selected key frames   220301 下采样globalMapKeyPoses相邻帧
            downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
            downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
                    //extract visualized and downsampled key frames 直接下采样提取关键帧
            for(int i = 0; i < globalMapKeyPosesDS->points.size(); ++i){
                int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
                *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                *globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            }
                    // downsample visualized points 220301 下采样globalMapKeyFrames,其也是可视化点
            downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
            downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);//220301 将下采样的地图帧转为ros类型消息
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubLaserCloudSurround.publish(cloudMsgTemp);//220301 发布地图
            
            globalMapKeyPoses->clear();
            globalMapKeyPosesDS->clear();
            globalMapKeyFrames->clear();//220301 发布之后，原来容器里面的内容释放？？？
            // globalMapKeyFramesDS->clear();
        }

        //220405can 回环检测可以消除漂移(drift)，通过ICP算法对比当前帧和之前帧是否匹配，如果匹配则进行图优化
        void loopClosureThread(){   //220305 回环进程？？？220405 对，回环检测进程

            if(loopClosureEnableFlag == false)
                return;
            
            ros::Rate rate(1);  //220301 速度１啥含义？？？
            while(ros::ok()){
                rate.sleep();
                //220405can 进行回环检测
                performLoopClosure();
            }
        }

        //220405can 判断是否进入回环在detectLoopClosure中进行，
        //220405can 判断条件是首尾之间的距离小于7米，并且时间相差30s以上
        bool detectLoopClosure(){   //220301 回环检测？？？？220405检测是否进入回环

            latestSurfKeyFrameCloud->clear();
            nearHistorySurfKeyFrameCloud->clear();
            nearHistorySurfKeyFrameCloudDS->clear();    //220302 回环检测开始之前，把所有容器清空？？？？
            //220306can 资源分配时初始化
            //220306can 在互斥量被析构前不解锁           
            std::lock_guard<std::mutex> lock(mtx);
            // find the closest history key frame
            std::vector<int> pointSearchIndLoop;
            std::vector<float> pointSearchSqDisLoop;
            kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
            //220306can
            // 进行半径historyKeyframeSearchRadius内的邻域搜索，
            // currentRobotPosPoint：需要查询的点，
            // pointSearchIndLoop：搜索完的邻域点对应的索引
            // pointSearchSqDisLoop：搜索完的每个邻域点与当前点之间的欧式距离
            // 0：返回的邻域个数，为0表示返回全部的邻域点
            //220306can
            //220405can 查找当前点7米范围内是否有之前已经采样的点(historyKeyframeSearchRadius在utilty.h中定义７米)
            kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

            closestHistoryFrameID = -1;//220302 找到闭环检测历史帧？？？？给历史帧赋值ｉｄ号？？？ 这里的历史帧，理解为相邻点
            for(int i = 0; i < pointSearchIndLoop.size(); ++i){
                int id = pointSearchIndLoop[i];
                //220405can 时间相差30s以上
                if(abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0){// 220306can 两个时间差值大于30秒
                    closestHistoryFrameID = id;
                    break;
                }
            }
            if(closestHistoryFrameID == -1){    //220306can 找到的点和当前时间上没有超过30秒的
                return false;
            }
            //save latest key frames 220302 保存在新的关键帧(点)
            latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
            //220306can 点云的xyz坐标进行坐标系变换(分别绕xyz轴旋转)
            *latestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]); //220302 为啥面点云容器添加的是边角点云？？？
            *latestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
            // 220306can
            // latestSurfKeyFrameCloud中存储的是下面公式计算后的index(intensity):
            // thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
            // 滤掉latestSurfKeyFrameCloud中index<0的点??? index值会小于0?(个人也没有确认答案)
            // 220306can
            pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
            int cloudSize = latestSurfKeyFrameCloud->points.size();
            for(int i = 0; i < cloudSize; ++i){
                //220306can
                // intensity不小于0的点放进hahaCloud队列
                // 初始化时intensity是-1，滤掉那些点
                // 220306can
                if((int)latestSurfKeyFrameCloud->points[i].intensity >= 0){
                    hahaCloud->push_back(latestSurfKeyFrameCloud->points[i]);   //这个容器存的帧点云接下来是用来干啥？？？
                }
            }
            latestSurfKeyFrameCloud->clear();
            *latestSurfKeyFrameCloud = *hahaCloud;
                //save history near key frames 220302 保存的是历史相邻帧（点）信息
                //220306can historyKeyframeSearchNum在utility.h中定义为25，前后25个点进行变换
            for(int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j){
                if(closestHistoryFrameID + j < 0 || closestHistoryFrameID + j > latestFrameIDLoopCloure)
                    continue;
                //220306can 要求closestHistoryFrameID + j在0到cloudKeyPoses3D->points.size()-1之间,不能超过索引
                *nearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[closestHistoryFrameID+j],&cloudKeyPoses6D->points[closestHistoryFrameID+j]);
                *nearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[closestHistoryFrameID+j], &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
            }
            //220306can  下采样滤波减少数据量
            downSizeFilterHistoryKeyFrames.setInputCloud(nearHistorySurfKeyFrameCloud);
            downSizeFilterHistoryKeyFrames.filter(*nearHistorySurfKeyFrameCloudDS); //220302 这两个下采样函数啥意思？？？
            //publish history near key frames 220302 发布历史相邻帧（点）信息
            if(pubHistoryKeyFrames.getNumSubscribers() != 0){
                sensor_msgs::PointCloud2 cloudMsgTemp;
                pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
                cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                cloudMsgTemp.header.frame_id = "/camera_init";
                pubHistoryKeyFrames.publish(cloudMsgTemp);
            }

            return true;
        }

        //220405can 如果检测到回环之后，接着进行ICP匹配，然后进行图优化
        void performLoopClosure(){  //220305 回环具体动作？220405 这里是回环检测的全流程
            
            if(cloudKeyPoses3D->points.empty() == true)
                return;
            //try to find close key frame if there are any 220303 从多帧中找到相邻帧
            //220405can 尝试去寻找回环点
            if(potentialLoopFlag == true){

                if(detectLoopClosure() == true){//220303 查找一些足够久足够近的帧进行回环闭合
                    potentialLoopFlag = true;   //find some key frames that is old enough or close enough for loop closure
                    timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
                }
                if(potentialLoopFlag == false)
                    return;
            }
            //reset the flag first no matter icp successes or not  无论进行icp成功与否，都把flag重置
            potentialLoopFlag = false;
            //icp　setting 220303 icp标准流程，具体函数含义是啥？类似下采样标准操作，百度一下找接口解析
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(100);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);//220306can 设置RANSAC运行次数
            //Align clouds
            //220405can 匹配当前帧点云和之前的历史点云
            icp.setInputSource(latestSurfKeyFrameCloud);
            //220306can 使用detectLoopClosure()函数中下采样刚刚更新nearHistorySurfKeyFrameCloudDS
            icp.setInputTarget(nearHistorySurfKeyFrameCloudDS);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);//220306can 进行icp点云对齐
            //220306can 为什么匹配分数高直接返回???分数高代表噪声太多
            if(icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
                return;
            //publish corrected cloud
            //220306can 以下在点云icp收敛并且噪声量在一定范围内进行
            if(pubIcpKeyFrames.getNumSubscribers() != 0){
                pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
                    //220306can icp.getFinalTransformation()的返回值是Eigen::Matrix<Scalar, 4, 4>
                pcl::transformPointCloud(*latestSurfKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());
                sensor_msgs::PointCloud2 cloudMsgTemp;
                pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
                cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                cloudMsgTemp.header.frame_id = "/camera_init";
                //220405can 发布的是校正后的点云
                pubIcpKeyFrames.publish(cloudMsgTemp);
            }

            // get pose constraint 220303 获取位置约束

            float x, y, z, roll, pitch, yaw;
            Eigen::Affine3f correctionCameraFrame;
            correctionCameraFrame = icp.getFinalTransformation();// get transformation in camera frame (because points are in camera frame)
                    //220306can 得到平移和旋转的角度
            pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);    //220303 注意函数参数顺序
            Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z, x, y, yaw, roll, pitch);   //220303 注意函数参数顺序
            //transform from world origin to wrong pose 220303 转换到错误位置？？意义是啥？？？？
            Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
            //transform from world origin to corrected pose 220303 转换到纠正位置，和上面的有啥配合作用？？？？如下关系
            Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;   //pre-multiplying -> successive rotation about a fixed frame 220303为啥这么算？？
            pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw),Point3(x, y, z));  //220303 函数以及对应的参数含义？？？
            gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);
            gtsam::Vector Vector6(6);
            float noiseScore = icp.getFitnessScore();   //220303 噪声指标？？？？
            Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
            constraintNoise = noiseModel::Diagonal::Variances(Vector6); //220303这个变量的作用是啥？？？

            // add constraints 220303 添加约束

            std::lock_guard<std::mutex> lock(mtx);
            gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise)); //220303 函数含义是啥？？？
            isam->update(gtSAMgraph);
            isam->update();
            gtSAMgraph.resize(0);

            aLoopIsClosed = true;   //220303 表示回环检测成功
        }

        Pose3 pclPointTogtsamPose3(PointTypePose thisPoint){    // camera frame to lidar frame 220303 相机的frame转到雷达frame
            return Pose3(Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll), double(thisPoint.pitch)),
                                Point3(double(thisPoint.z), double(thisPoint.x), double(thisPoint.y)));//220303 这个函数是作用是啥？里面的参数有啥要求？？？
        }

        Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointTypePose thisPoint){   //camera frame to lidar frame
                return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y, thisPoint.yaw, thisPoint.roll, thisPoint.pitch);   //这个转换函数返回的是啥？和上面的函数不同意义在哪里？？
        }

        //220405can 如果使能了回环检测，则添加过去50个关键帧，
        //如果没有使能回环检测，则添加离当前欧式距离最近的50个关键帧，然后拼接成点云
        void extractSurroundingKeyFrames(){ //220303 提取环境关键帧(点)

            if(cloudKeyPoses3D->points.empty() == true)
                return;
            
            //220405can 若使用回环检测
            if(loopClosureEnableFlag == true){
                //only use recent key poses for graph building 220303 仅使用最近的关键位置进行建图
                //220306can loopClosureEnableFlag 这个变量另外只在loopthread这部分中有用到
                //220405can 添加最近的关键位姿到图
                if(recentCornerCloudKeyFrames.size() < surroundingKeyframeSearchNum){// queue is not full (the beginning of mapping or a loop is just closed) 220303　队列不满（由于地图刚开始或者第一圈刚闭合？？？如何消除二义性？？？（不知道如何消除））这里理解为保存的点云数量不足
                    //220306can recentCornerCloudKeyFrames保存的点云数量太少，则清空后重新塞入新的点云直至数量够
                    //clear recent key frames queue 220303 清除最近的关键帧序列
                    recentCornerCloudKeyFrames.clear();
                    recentSurfCloudKeyFrames.clear();
                    recentOutlierCloudKeyFrames.clear();
                    int numPoses = cloudKeyPoses3D->points.size();
                    for(int i = numPoses - 1; i >= 0; --i){
                        //220306can cloudKeyPoses3D的intensity中存的是索引值?
                        //220306can 保存的索引值从1开始编号；
                        int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity; //220303 把反射强度作为ｉｄ？？？
                        PointTypePose thisTransformtion = cloudKeyPoses6D->points[thisKeyInd];
                        updateTransformPointCloudSinCos(&thisTransformtion);    //220303 更新点云点的sin,cos,那么更新这些参数，作用是啥？？？？
                        //extract surrounding map   220303 提取环境地图
                        //220306can 依据上面得到的变换thisTransformation，对cornerCloudKeyFrames，surfCloudKeyFrames，surfCloudKeyFrames
                        //220306can 进行坐标变换
                        //220405can 提取过去50个关键帧
                        recentCornerCloudKeyFrames.push_front(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                        recentSurfCloudKeyFrames.push_front(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                        recentOutlierCloudKeyFrames.push_front(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                        if(recentCornerCloudKeyFrames.size() >= surroundingKeyframeSearchNum)   //220303 　容器满了，就不再添加？？？
                            break;
                    }
                }else{  //queue is full, pop the oldest key frame and push the latest key frame 220303 队列满了，去除最过去的点
                    //220306can recentCornerCloudKeyFrames中点云保存的数量较多
                    //220306can pop(去除)队列最前端的一个，再push（添加）后面一个
                    if(latestFrameID != cloudKeyPoses3D->points.size() - 1){ // if the robot is not moving, no need to update recent frames  220303如果机器人没有移动，则不需要更新最近帧
                    
                        recentCornerCloudKeyFrames.pop_front();
                        recentSurfCloudKeyFrames.pop_front();
                        recentOutlierCloudKeyFrames.pop_front();
                        //220306can 为什么要把recentCornerCloudKeyFrames最前面第一个元素弹出?
                        //220306can (1个而不是好几个或者是全部)?    (个人没有理解这里的疑惑点)
                        //push latest scan to the end of queue  220303 添加最新的一帧到队列的尾部
                        //220405can 弹出队列中时间最久的帧，添加最新的帧到队列
                        latestFrameID = cloudKeyPoses3D->points.size() - 1;
                        PointTypePose thisTransformation = cloudKeyPoses6D->points[latestFrameID];
                        updateTransformPointCloudSinCos(&thisTransformation);
                        recentCornerCloudKeyFrames.push_back(transformPointCloud(cornerCloudKeyFrames[latestFrameID]));
                        recentSurfCloudKeyFrames.push_back(transformPointCloud(surfCloudKeyFrames[latestFrameID]));
                        recentOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[latestFrameID]));
                    }
                }
                //220405can 拼接为点云
                for(int i = 0; i<recentCornerCloudKeyFrames.size(); ++i){
                    //220306can 两个pcl::PointXYZI相加?
                    //220306can 注意这里把recentOutlierCloudKeyFrames也加入到了laserCloudSurfFromMap
                    *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
                    *laserCloudSurfFromMap += *recentSurfCloudKeyFrames[i];
                    *laserCloudSurfFromMap += *recentOutlierCloudKeyFrames[i];  // 220303 地图的面点云也添加外部类型点云？？？是的
                }
            }else{  //220405 回环检测没有启动时
                //220306can 下面这部分是没有闭环的代码
                surroundingKeyPoses->clear();
                surroundingKeyPosesDS->clear();//220303 为啥这两个环境关键位姿清空？？？？
                // extract all the nearby key poses and downsample them 提取相邻的关键位姿，然后对其进行下采样
                //220405can 查找当前pose 50米内的姿态
                kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
                // 220306can
                // 进行半径surroundingKeyframeSearchRadius内的邻域搜索，
                // currentRobotPosPoint：需要查询的点，
                // pointSearchInd：搜索完的邻域点对应的索引
                // pointSearchSqDis：搜索完的每个领域点点与传讯点之间的欧式距离
                // 0：返回的邻域个数，为0表示返回全部的邻域点
                // 220306can
                kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis, 0);    //220304 这几个参数有没有包含检索后返回值？？？
                for(int i = 0; i < pointSearchInd.size(); ++i)
                    surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);  //220304 kd树返回检索点填入？？？
                downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
                downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);//220304 下采样操作
                // delete key frames that are not in surrounding region 220304 删除非周围区域的关键帧
                int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();
                for(int i = 0; i < surroundingExistingKeyPosesID.size(); ++i){  //220304 这两个for循环的作用是啥？
                    bool existingFlag = false;
                    for( int j = 0; j < numSurroundingPosesDS; ++j){
                    //220306can 双重循环，不断对比surroundingExistingKeyPosesID[i]和surroundingKeyPosesDS的点的index
                    //220306can 如果能够找到一样的，说明存在相同的关键点(因为surroundingKeyPosesDS从cloudKeyPoses3D中筛选而来)
                        if(surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity){
                            existingFlag = true;
                            break;
                        }
                    }
                    if(existingFlag == false){   //220304 为啥条件存在时会清除这些点？？？这里表示上面的对比遍历，发现没有匹配上的点，就清除这些容器的点
                        //220306can 如果surroundingExistingKeyPosesID[i]对比了一轮的已经存在的关键位姿的索引后（intensity保存的就是size()）
                        //220306can 没有找到相同的关键点，那么把这个点从当前队列中删除
                        //220306can 否则的话，existingFlag为true，该关键点就将它留在队列中
                        surroundingExistingKeyPosesID.erase(surroundingExistingKeyPosesID.begin() + i);
                        surroundingCornerCloudKeyFrames.erase(surroundingCornerCloudKeyFrames.begin() + i);
                        surroundingSurfCloudKeyFrames.erase(surroundingSurfCloudKeyFrames.begin() + i);
                        surroundingOutlierCloudKeyFrames.erase(surroundingOutlierCloudKeyFrames.begin() + i);
                        --i;
                    }
                }
                //220306can 上一个两重for循环主要用于删除数据，此处的两重for循环用来添加数据
                //add new key frames that are not in calculated existing key frames 220304 添加新的帧
                //220405can 添加关键帧
                for(int i = 0; i < numSurroundingPosesDS; ++i){
                    bool existingFlag = false;
                    for(auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter){
                        //220306can *iter就是不同的cloudKeyPoses3D->points.size(),
                        //220306can 把surroundingExistingKeyPosesID内没有对应的点放进一个队列里
                        //220306can 这个队列专门存放周围存在的关键帧，但是和surroundingExistingKeyPosesID的点没有对应的，也就是新的点
                        if((*iter) == (int)surroundingKeyPosesDS->points[i].intensity){//220304 根据ｉｄ来判断是否检索过的？？？？
                            existingFlag = true;
                            break;
                        }
                    }
                    if(existingFlag == true){
                        continue;
                    }else{  //220304 添加新帧
                        int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                        PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                        updateTransformPointCloudSinCos(&thisTransformation);
                        surroundingExistingKeyPosesID.push_back(thisKeyInd);
                        surroundingCornerCloudKeyFrames.push_back(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                        surroundingSurfCloudKeyFrames.push_back(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                        surroundingOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                    }
                }
                //220405can 拼接点云
                for(int i = 0; i < surroundingExistingKeyPosesID.size(); ++i){
                    *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
                    *laserCloudSurfFromMap += *surroundingSurfCloudKeyFrames[i];
                    *laserCloudSurfFromMap += *surroundingOutlierCloudKeyFrames[i];
                }                
            }
            //220306can 进行两次下采样
            //220306can 最后的输出结果是laserCloudCornerFromMapDS和laserCloudSurfFromMapDS            
            //downsample the surrounding corner key frames (or map) 下采样角点帧或者地图点
            //220405can 下采样角特征和面特征
            downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
            downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);    //220304 下采样两步样式
            laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();
            //downsample the surrounding surf key frames (or map) 220304 下采样面点
            downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
            downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
            laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
        }

        void downsampleCurrentScan(){   //220304 下采样当前扫描点云
            
            laserCloudCornerLastDS->clear();
            downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
            downSizeFilterCorner.filter(*laserCloudCornerLastDS);
            laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();

            laserCloudSurfLastDS->clear();
            downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
            downSizeFilterSurf.filter(*laserCloudSurfLastDS);
            laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();

            laserCloudOutlierLastDS->clear();
            downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);
            downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);
            laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size();

            laserCloudSurfTotalLast->clear();
            laserCloudSurfTotalLastDS->clear();
            *laserCloudSurfTotalLast += *laserCloudSurfLastDS;
            *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
            downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
            downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
            laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
        }

        //220405can 角特征优化
        void cornerOptimization(int iterCount){     //220305 （边沿点）角点（凸点）优化？？corner,surf,outlier三种类型的点具体如何区分？220405分布是角、面、离群点
            
            updatePointAssociateToMapSinCos();
            for(int i = 0; i < laserCloudCornerLastDSNum; i++){
                pointOri = laserCloudCornerLastDS->points[i];
                //220306can 进行坐标变换,转换到全局坐标中去（世界坐标系）
                //220306can pointSel:表示选中的点，point select
                //220306can 输入是pointOri，输出是pointSel
                pointAssociateToMap(&pointOri, &pointSel);
                //220306can
                // 进行5邻域搜索，
                // pointSel为需要搜索的点，
                // pointSearchInd搜索完的邻域对应的索引
                // pointSearchSqDis 邻域点与查询点之间的距离
                //220306can
                //220405can 查找最近的5个点,
                kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
                
                //220306can 只有当最远的那个邻域点的距离pointSearchSqDis[4]小于1m时才进行下面的计算
                //220306can 以下部分的计算是在计算点集的协方差矩阵，Zhang Ji的论文中有提到这部分
                //220405can 找到的点，距离小于1m
                if(pointSearchSqDis[4] < 1.0){  //220304 这个判断条件啥意思？？？？可以看上一行解析
                    //220306can 先求5个样本的平均值
                    float cx = 0, cy = 0, cz = 0;
                    for(int j = 0; j < 5; ++j){
                        cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                        cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                        cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                    }
                    //220405can 5个点的质心
                    cx /= 5; cy /= 5; cz /= 5;  //220304 这几个含义是啥？为啥一定除以５？由于计算的是５帧点云数据
                    //220306can 下面在求矩阵matA1=[ax,ay,az]^t*[ax,ay,az]
                    //220306can 更准确地说应该是在求协方差matA1
                    float a11 = 0, a12 = 0, a13 = 0, a22 = 0,a23 = 0, a33 = 0;
                    for(int j = 0; j < 5; j++){
                        //220306can ax代表的是x-cx,表示均值与每个实际值的差值，求取5个之后再次取平均，得到matA1
                        float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                        float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                        float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                        a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                        a22 + ay * ay; a23 += ay * az;
                        a33 += az * az;         //220305 这个循环里面的公式含义是啥？？每个变量代表什么意思？？？可以看上一行注释
                    }
                    //220405can 协方差
                    a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                    matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                    matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                    matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;  //220305　构建这个对角矩阵啥作用？？？
                    //220306can 求正交阵的特征值和特征向量
                    //220306can 特征值：matD1，特征向量：matV1中
                    cv::eigen(matA1, matD1, matV1);
                    //220306can 边缘：与较大特征值相对应的特征向量代表边缘线的方向（一大两小，大方向）
                    //220306can 以下这一大块是在计算点到边缘的距离，最后通过系数s来判断是否距离很近
                    //220306can 如果距离很近就认为这个点在边缘上，需要放到laserCloudOri中
                    if(matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)){  //220305 这个判断条件出于什么理由提出的？？？ 解析者写矩阵内容，或许有答案

                        float x0 = pointSel.x;
                        float y0 = pointSel.y;
                        float z0 = pointSel.z;
                        float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                        float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                        float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                        float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                        float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                        float z2 = cz - 0.1 * matV1.at<float>(0, 2);
                        //220305 计算的是啥值？？？或者说这些值含义是啥？？？
                        //220306can
                        // 这边是在求[(x0-x1),(y0-y1),(z0-z1)]与[(x0-x2),(y0-y2),(z0-z2)]叉乘得到的向量的模长
                        // 这个模长是由0.2*V1[0]和点[x0,y0,z0]构成的平行四边形的面积
                        // 因为[(x0-x1),(y0-y1),(z0-z1)]x[(x0-x2),(y0-y2),(z0-z2)]=[XXX,YYY,ZZZ],
                        // [XXX,YYY,ZZZ]=[(y0-y1)(z0-z2)-(y0-y2)(z0-z1),-(x0-x1)(z0-z2)+(x0-x2)(z0-z1),(x0-x1)(y0-y2)-(x0-x2)(y0-y1)]
                        //220306can （个人的线性代数知识需要重新看一下）
                        float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                        * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                        + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                        * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                        + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                        * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
                        //220306can l12表示的是0.2*(||V1[0]||)
                        //220306can 也就是平行四边形一条底的长度  
                        float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
                        //220306can 求叉乘结果[la',lb',lc']=[(x1-x2),(y1-y2),(z1-z2)]x[XXX,YYY,ZZZ]
                        //220306can [la,lb,lc]=[la',lb',lc']/a012/l12
                        //220306can LLL=[la,lb,lc]是0.2*V1[0]这条高上的单位法向量。||LLL||=1；
                        float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 -x2)*(y0 - y1))
                                    + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)))/ a012 / l12;

                        float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    -(z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
                        
                        float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    +(y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
                        //220306can 计算点pointSel到直线的距离  (这条直线具体指的是？？个人没有理解)
                        //220306can 这里需要特别说明的是ld2代表的是点pointSel到过点[cx,cy,cz]的方向向量直线的距离
                        float ld2 = a012 / l12;
                        //220306can 如果在最理想的状态的话，ld2应该为0，表示点在直线上
                        //220306can 最理想状态s=1；
                        float s = 1 -0.9 * fabs(ld2);   //220305 表示的是啥含义？？？描述点离线的偏移程度
                        //220306can coeff代表系数的意思
                        //220306can coff用于保存距离的方向向量
                        coeff.x = s * la;
                        coeff.y = s * lb;
                        coeff.z = s * lc;
                        //220306can intensity本质上构成了一个核函数，ld2越接近于1，增长越慢
                        //220306can intensity=(1-0.9*ld2)*ld2=ld2-0.9*ld2*ld2
                        coeff.intensity = s * ld2;  //220305 这个点表示的是啥点？？？根据参考资料，这里可能并不是描述点，而是一个系数结构
                        
                        //220306can 所以就应该认为这个点是边缘点
                        //220306can s>0.1 也就是要求点到直线的距离ld2要小于1m
                        //220306can s越大说明ld2越小(离边缘线越近)，这样就说明点pointOri在直线上
                        if(s > 0.1){
                            laserCloudOri->push_back(pointOri); //220305 这个容器里面放的是啥点？？？
                            coeffSel->push_back(coeff);
                        }
                    }
                }
            }
        }

        //220405can 面特征优化
        void surfOptimization(int iterCount){   //220305 面优化
            updatePointAssociateToMapSinCos();
            for(int i = 0; i < laserCloudSurfTotalLastDSNum; i++){
                pointOri = laserCloudSurfTotalLastDS->points[i];
                pointAssociateToMap(&pointOri, &pointSel);
                kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);   //220305 最近帧查找？？？

                if(pointSearchSqDis[4] < 1.0){  //220305 条件啥意思？
                    for(int j = 0; j < 5; j++){
                        matA0.at<float>(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                        matA0.at<float>(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                        matA0.at<float>(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                    }
                    //220306can
                    // matB0是一个5x1的矩阵
                    // matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
                    // matX0是3x1的矩阵
                    // 求解方程matA0*matX0=matB0
                    // 公式其实是在求由matA0中的点构成的平面的法向量matX0
                    //220306can
                    cv::solve(matA0, matB0, matX0, cv::DECOMP_QR); // 220305 A0 * X0 = B0,求解X0，也就是保存结果在矩阵matX0中

                    //220306can
                    // [pa,pb,pc,pd]=[matX0,pd]
                    // 正常情况下（见后面planeValid判断条件），应该是
                    // pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                    // pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                    // pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z = -1
                    // 所以pd设置为1
                    //220306can
                    float pa = matX0.at<float>(0, 0);
                    float pb = matX0.at<float>(1, 0);
                    float pc = matX0.at<float>(2, 0);
                    float pd = 1;

                    //220306can 对[pa,pb,pc,pd]进行单位化
                    float ps = sqrt(pa * pa + pb * pb + pc * pc);
                    pa /= ps; pb /= ps; pc /= ps; pd /= ps;
                    
                    //220306can 求解后再次检查平面是否是有效平面
                    bool planeValid = true;
                    for(int j = 0; j < 5; j++){
                        if(fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                                pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                                pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pb) > 0.2){ //220305 这里判断条件是啥含义？？？
                            planeValid = false;
                            break;
                        }
                    }

                    if(planeValid){
                        float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                        //220306can 后面部分相除求的是[pa,pb,pc,pd]与pointSel的夹角余弦值(两个sqrt，其实并不是余弦值)
                        //220306can 这个夹角余弦值越小越好，越小证明所求的[pa,pb,pc,pd]与平面越垂直
                        float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                                +pointSel.y * pointSel.y + pointSel.z * pointSel.z));   // 220305 这里计算的是什么变量？？意义在哪里？
                        
                        coeff.x = s * pa;
                        coeff.y = s * pb;
                        coeff.z = s * pc;
                        coeff.intensity = s * pd2;

                        //220306can 判断是否是合格平面，是就加入laserCloudOri
                        if(s > 0.1){    // 220305 这个条件根据是什么？表示点偏离线程度在接受范围，保存信息
                            laserCloudOri->push_back(pointOri);
                            coeffSel->push_back(coeff);
                        }
                    }
                }
            }
        }

        //220306can 这部分的代码是基于高斯牛顿法的优化，不是zhang ji论文中提到的基于L-M的优化方法
        //220306can 这部分的代码使用旋转矩阵对欧拉角求导，优化欧拉角，不是zhang ji论文中提到的使用angle-axis的优化
        bool LMOptimization(int iterCount){     //220305 高斯牛顿优化
            float srx = sin(transformTobeMapped[0]);
            float crx = cos(transformTobeMapped[0]);
            float sry = sin(transformTobeMapped[1]);
            float cry = cos(transformTobeMapped[1]);
            float srz = sin(transformTobeMapped[2]);
            float crz = cos(transformTobeMapped[2]);

            int laserCloudSelNum = laserCloudOri->points.size();
            //220306can laser cloud original 点云太少，就跳过这次循环
            if(laserCloudSelNum < 50){      //220305 这个５０依据是什么？
                return false;
            }

            cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));// 220305 相对上一行来说，是转置矩阵？
            cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));   //220305  这个矩阵作用又是什么？
            cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
            cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
            cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
            //220405can 遍历laserCloudSel
            for(int i = 0;  i < laserCloudSelNum; i++){
                pointOri = laserCloudOri->points[i];
                coeff = coeffSel->points[i];
                //220305 这几个计算的是什么值？有什么作用？
                //220306can 求雅克比矩阵中的元素，距离d对roll角度的偏导量即d(d)/d(roll)
                //220306can 更详细的数学推导参看wykxwyc.github.io
                float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                            + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                            + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z)*coeff.z;
                
                //220306can 同上，求解的是对pitch的偏导量
                float ary = ((cry*srx*srz - crz*sry)*pointOri.x
                            + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                            + ((-cry*crz - srx*sry*srz)*pointOri.x
                            +(cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;
                
                float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                            + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                            + ((sry*srz + cry*srz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

                /*220306can
                在求点到直线的距离时，coeff表示的是如下内容
                [la,lb,lc]表示的是点到直线的垂直连线方向，s是长度
                coeff.x = s * la;
                coeff.y = s * lb;
                coeff.z = s * lc;
                coeff.intensity = s * ld2;
                在求点到平面的距离时，coeff表示的是
                [pa,pb,pc]表示过外点的平面的法向量，s是线的长度
                coeff.x = s * pa;
                coeff.y = s * pb;
                coeff.z = s * pc;
                coeff.intensity = s * pd2;
                */
                matA.at<float>(i, 0) = arx;     //220305 不会和下面反光率冲突吗？不会，矩阵不一样
                matA.at<float>(i, 1) = ary;
                matA.at<float>(i, 2) = arz;

                //220306can 这部分是雅克比矩阵中距离对平移的偏导
                matA.at<float>(i, 3) = coeff.x;
                matA.at<float>(i, 4) = coeff.y;
                matA.at<float>(i, 5) = coeff.z;

                //220306can 残差项
                matB.at<float>(i, 0) = -coeff.intensity;    //220305 个人认为这里应该是(i,6)，不然就会覆盖上面的值，是吗？不是，这里是矩阵matB了
            }
            //220306can 将矩阵由matA转置生成matAt
            //220306can 先进行计算，以便于后边调用 cv::solve求解
            //220405can 最小二乘法
            cv::transpose(matA, matAt);
            matAtA = matAt * matA;
            matAtB = matAt * matB;

            //220306can
            // 利用高斯牛顿法进行求解，
            // 高斯牛顿法的原型是J^(T)*J * delta(x) = -J*f(x)
            // J是雅克比矩阵，这里是A，f(x)是优化目标，这里是-B(符号在给B赋值时候就放进去了)
            // 通过QR分解的方式，求解matAtA*matX=matAtB，得到解matX
            //220306can
            cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

            //220306can iterCount==0 说明是第一次迭代，需要初始化
            if(iterCount == 0){
                cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
                cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
                cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

                //220306can 对近似的Hessian矩阵求特征值和特征向量，
                cv::eigen(matAtA, matE, matV);
                matV.copyTo(matV2);

                isDegenerate = false;
                float eignTre[6] = {100, 100, 100, 100, 100, 100};  //220305 这个数据这样子初始化的依据是什么？接下来要来做什么？
                for(int i = 5; i>= 0; i--){ //220305 循环作用是什么？
                    if(matE.at<float>(0, i) < eignTre[i]){
                        for(int j = 0; j < 6; j++){
                            matV2.at<float>(i, j) = 0;
                        }
                        isDegenerate = true;
                    }else{
                        break;
                    }
                }
                matP = matV.inv() * matV2;                
            }

            if(isDegenerate){   //220305 表示生成？指的是什么生成？
                cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
                matX.copyTo(matX2);
                matX = matP * matX2;
            }
            //220405can 获取transform
            transformTobeMapped[0] += matX.at<float>(0, 0);
            transformTobeMapped[1] += matX.at<float>(1, 0);
            transformTobeMapped[2] += matX.at<float>(2, 0);
            transformTobeMapped[3] += matX.at<float>(3, 0);
            transformTobeMapped[4] += matX.at<float>(4, 0);
            transformTobeMapped[5] += matX.at<float>(5, 0);
            
            float deltaR = sqrt(
                                pow(pcl::rad2deg(matX.at<float>(0, 0)),2) +
                                pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                                pow(pcl::rad2deg(matX.at<float>(2, 0)),2));
            float deltaT = sqrt(
                pow(matX.at<float>(3, 0) * 100, 2) +
                pow(matX.at<float>(4, 0) * 100, 2) +
                pow(matX.at<float>(5, 0) * 100, 2));
            //220306can 旋转或者平移量足够小就停止这次迭代过程
            //220405 这里认为优化收敛了，则停止迭代
            if(deltaR < 0.05 && deltaT < 0.05){
                return true;
            }
            return false;
        }

        //220405can 通过最小二乘法，添加当前的扫描帧到map
        void scan2MapOptimization(){    //220305 扫面转为地图时的优化
            //220306can laserCloudCornerFromMapDSNum是extractSurroundingKeyFrames()函数最后降采样得到的coner点云数
            //220306can laserCloudSurfFromMapDSNum是extractSurroundingKeyFrames()函数降采样得到的surface点云数
            //220405can 根据周围关键帧点云创建kdtree（角特征点云、面特征点云）
            if(laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100){
                //220306can laserCloudCornerFromMapDS和laserCloudSurfFromMapDS的来源有2个：
                //220306can 当有闭环时，来源是：recentCornerCloudKeyFrames，没有闭环时，来源surroundingCornerCloudKeyFrames
                kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
                kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

                for(int iterCount = 0; iterCount < 10; iterCount++){    // 220305  这里10的理由？下一行解析
                    //220306can 用for循环控制迭代次数，最多迭代10次
                    laserCloudOri->clear();
                    coeffSel->clear();
                    //220405can 角和面特征计算距离
                    cornerOptimization(iterCount);
                    surfOptimization(iterCount);
                    //220405can LM优化
                    if(LMOptimization(iterCount) == true)
                        break;
                }
                //220306can 迭代结束更新相关的转移矩阵
                transformUpdate();
            }
        }

        //220405can 通过前后2帧进行优化，保存优化后的位姿
        void saveKeyFramesAndFactor(){  //220305 保存关键帧和元素

            currentRobotPosPoint.x = transformAftMapped[3];
            currentRobotPosPoint.y = transformAftMapped[4];
            currentRobotPosPoint.z = transformAftMapped[5];
            //220405can 当前帧和之前帧的距离小于0.3米
            bool saveThisKeyFrame = true;
            if(sqrt((previousRobotPosPoint.x - currentRobotPosPoint.x)*(previousRobotPosPoint.x - currentRobotPosPoint.x)
                    +(previousRobotPosPoint.y - currentRobotPosPoint.y)*(previousRobotPosPoint.y - currentRobotPosPoint.y)
                    +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) < 0.3){
                saveThisKeyFrame = false;
            }
            //220405can saveThisKeyFrame为false，并且cloudKeyPoses3D不为空
            if(saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())    //220305 不需要保存操作并且容器里面为空时，直接中断该函数
                    return;

            previousRobotPosPoint = currentRobotPosPoint;

            //update gtsam graph 220305 更新gtsam 图
            //220405can 把当前pose加入grsam graph

            if(cloudKeyPoses3D->points.empty()){
                //220306can
                // static Rot3 	RzRyRx (double x, double y, double z),Rotations around Z, Y, then X axes
                // RzRyRx依次按照z(transformTobeMapped[2])，y(transformTobeMapped[0])，x(transformTobeMapped[1])坐标轴旋转
                // Point3 (double x, double y, double z)  Construct from x(transformTobeMapped[5]), y(transformTobeMapped[3]), and z(transformTobeMapped[4]) coordinates. 
                // Pose3 (const Rot3 &R, const Point3 &t) Construct from R,t. 从旋转和平移构造姿态
                // NonlinearFactorGraph增加一个PriorFactor因子
                //220306can （个人这里还没有理解）
                gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                                        Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])),priorNoise));
                //220306can initialEstimate的数据类型是Values,其实就是一个map，这里在0对应的值下面保存了一个Pose3
                initialEstimate.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                        Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
                for(int i = 0; i < 6; ++i)
                    transformLast[i] = transformTobeMapped[i];  //220305 这个赋值替换是什么？
            }
            else{//220305 这里的逻辑含义是什么？
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
                                                    Point3(transformLast[5], transformLast[3], transformLast[4]));
                gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                    Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));
                //220306can 构造函数原型:BetweenFactor (Key key1, Key key2, const VALUE &measured, const SharedNoiseModel &model)
                gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
                initialEstimate.insert(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                    Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4])));
            }

            //update isam 220305 更新isam ．　个人没理解isam是什么。一个图优化的库gtsam(平滑和建图)
            //gtsam与g2o区别：g2o采用稀疏矩阵的方式求解一个非线性优化问题；gtsam采用因子图（factor graphs）和贝叶斯网络（Bayes networks）的方式最大化后验概率
            //220306can
            // gtsam::ISAM2::update函数原型:
            // ISAM2Result gtsam::ISAM2::update	(	const NonlinearFactorGraph & 	newFactors = NonlinearFactorGraph(),
            // const Values & 	newTheta = Values(),
            // const std::vector< size_t > & 	removeFactorIndices = std::vector<size_t>(),
            // const boost::optional< FastMap< Key, int > > & 	constrainedKeys = boost::none,
            // const boost::optional< FastList< Key > > & 	noRelinKeys = boost::none,
            // const boost::optional< FastList< Key > > & 	extraReelimKeys = boost::none,
            // bool 	force_relinearize = false )	
            // gtSAMgraph是新加到系统中的因子
            // initialEstimate是加到系统中的新变量的初始点
            //220306can
            //220405can 更新isam
            isam->update(gtSAMgraph, initialEstimate);
            //220306can update 函数为什么需要调用两次？(个人也不懂为啥)
            isam->update();

            //220306can 删除内容?
            gtSAMgraph.resize(0);
            initialEstimate.clear();


            // save key poses 220305 保存关键帧

            PointType thisPose3D;
            PointTypePose thisPose6D;
            Pose3 latestEstimate;

            //220306can Compute an estimate from the incomplete linear delta computed during the last update.
            isamCurrentEstimate = isam->calculateEstimate();
            latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

            thisPose3D.x = latestEstimate.translation().y();
            thisPose3D.y = latestEstimate.translation().z();      
            thisPose3D.z = latestEstimate.translation().x();    // 220305 这里取得都是交叉变换赋值，有啥特殊含义？？
            thisPose3D.intensity = cloudKeyPoses3D->points.size();  //this can be used as index 220305 作为序号
            cloudKeyPoses3D->push_back(thisPose3D);

            thisPose6D.x = thisPose3D.x;
            thisPose6D.y = thisPose3D.y;
            thisPose6D.z = thisPose3D.z;
            thisPose6D.intensity = thisPose3D.intensity;    //this can be used as index
            thisPose6D.roll = latestEstimate.rotation().pitch();
            thisPose6D.pitch = latestEstimate.rotation().yaw();
            thisPose6D.yaw = latestEstimate.rotation().roll();  // in camera frame 220305 这几个角度交替互换，是啥含义呢？？
            thisPose6D.time = timeLaserOdometry;
            cloudKeyPoses6D->push_back(thisPose6D);

            //save updated transform 保存更新转换

            if(cloudKeyPoses3D->points.size() > 1){ //220305 只要有点，都要进行更新
                transformAftMapped[0] = latestEstimate.rotation().pitch();
                transformAftMapped[1] = latestEstimate.rotation().yaw();
                transformAftMapped[2] = latestEstimate.rotation().roll();
                transformAftMapped[3] = latestEstimate.translation().y();
                transformAftMapped[4] = latestEstimate.translation().z();
                transformAftMapped[5] = latestEstimate.translation().x();// 220305 这里的坐标更新，这么赋值理由是什么？

                for(int i = 0; i < 6; ++i){     //220305 这里更新的是什么？
                    transformLast[i] = transformAftMapped[i];
                    transformTobeMapped[i] = transformAftMapped[i];
                }
            }

            pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());   // 220305 定义变量
            pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(new pcl::PointCloud<PointType>());

            //220306can PCL::copyPointCloud(const pcl::PCLPointCloud2 &cloud_in,pcl::PCLPointCloud2 &cloud_out )  
            pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame); //220305 拷贝点云,将下采样的拷贝出来
            pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);
            pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);
            //220405can 保存点云
            cornerCloudKeyFrames.push_back(thisCornerKeyFrame);     //220305 把获取的点云保存到对应的容器里面
            surfCloudKeyFrames.push_back(thisSurfKeyFrame);
            outlierCloudKeyFrames.push_back(thisOutlierKeyFrame);
        }

        //220405can 回环检测如果成功，会设置aLoopIsClosed为true，才会进行这一步。将isam优化后的位姿替换之前的位姿
        void correctPoses(){    //220305 纠正姿态
            if(aLoopIsClosed == true){   //220305 判断是否闭合一圈点云？220405 true表示回环检测成功意思
                recentCornerCloudKeyFrames.clear();
                recentSurfCloudKeyFrames.clear();
                recentOutlierCloudKeyFrames.clear();
                //update key pose   220305 更新位姿
                //220405can 将isam优化后的位姿替换之前的位姿
                int numPoses = isamCurrentEstimate.size();
                for(int i = 0; i < numPoses; ++i){
                    cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().y();
                    cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().z();
                    cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().x();

                    cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                    cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                    cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                    cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                    cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
                    cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<Pose3>(i).rotation().roll(); //220305 这里的交叉赋值问题和上面提及一致，这么写的缘由是什么？220405 将优化后的位姿替换之前的位姿
                }

                aLoopIsClosed = false;  //220405 重置参数，为下一次回环检测做准备
            }
        }

        void clearCloud(){  //220305 清除点云
            laserCloudCornerFromMap->clear();
            laserCloudSurfFromMap->clear();
            laserCloudCornerFromMapDS->clear();
            laserCloudSurfFromMapDS->clear();
        }

        void run(){ //220305 该模块的运行总集合
            //220405can 
            if(newLaserCloudCornerLast && std::abs(timeLaserCloudCornerLast - timeLaserOdometry) < 0.005 &&
                newLaserCloudSurfLast && std::abs(timeLaserCloudSurfLast - timeLaserOdometry) < 0.005 &&
                newLaserCloudOutlierLast && std::abs(timeLaserCloudOutlierLast - timeLaserOdometry) < 0.005 &&
                newLaserOdometry)    //220305 这里的时间判断依据是什么？数据流的时间间隔
            {

                newLaserCloudCornerLast = false; newLaserCloudSurfLast = false; newLaserCloudOutlierLast = false; newLaserOdometry = false;

                std::lock_guard<std::mutex> lock(mtx);
                //220405can 
                if(timeLaserOdometry - timeLastProcessing >= mappingProcessInterval){   //220305 odom时间点减去处理时间点大于建图过程间隙时间
                    //220405can 
                    timeLastProcessing = timeLaserOdometry;
                    //220405can 转换到map坐标系
                    transformAssociateToMap();  //220305    把相关的转到地图中
                    //220405can 提取周围关键帧
                    extractSurroundingKeyFrames();  //220305    提取环境中关键帧
                    //220405can 下采样当前帧
                    downsampleCurrentScan();    //220305    下采样当前扫描点
                    //220405can scan到map优化
                    //220306can 当前扫描进行边缘优化，图优化以及进行LM优化的过程
                    scan2MapOptimization();     //220305    扫描转地图的优化
                    //220405can 保存关键帧和因子
                    saveKeyFramesAndFactor();   //220305    保存关键元素和帧
                    //220405can 校正位姿
                    correctPoses();             //220305    纠正位姿
                    //220405can 发布坐标转换
                    publishTF();                //220305    发布ＴＦ
                    //220405can  发布关键帧和姿态
                    publishKeyPosesAndFrames(); //220305    发布关键位姿和帧
                    //220405can  清除点云
                    clearCloud();               //220305    清理点云
                }
            }
        }//220405can lego-loam 加入了回环检测，同时通过scan-2-map的方式生成点云地图
};


int main(int argc,char** argv)
{   
    ros::init(argc, argv, "logo_loam");

    ROS_INFO("\033[1;32m----->\033[0m Map Optimization Started.");

    mapOptimization MO;

    //220306can std::thread 构造函数，将MO作为参数传入构造的线程中使用
    //220306can 进行闭环检测与闭环的功能
    //220405can 回环检测
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    //220306can 该线程中进行的工作是publishGlobalMap(),将数据发布到ros中，可视化
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::Rate rate(200);
    while(ros::ok())
    {
        ros::spinOnce();
        //220405can 主进程
        MO.run();

        rate.sleep();
    }

    loopthread.join();
    visualizeMapThread.join();
    
    return 0;
}