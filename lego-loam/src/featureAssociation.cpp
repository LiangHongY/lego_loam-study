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
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#include"utility.h"

//220403 这个模块作用是进行特征关联

//220405can 在imageProjection.cpp中，进行了地面分割，对非地面点进行聚类，最后得到地面点云和分割点云
//这个模块，就是对分割点云进行特征提取。提取特征的目的是进行点云配准，从而获取当前的位姿


class FeatureAssociation{

private:
    //220308 这里声明的变量含义，没搞清楚．需要根据下面的使用情况进行反向推导获取
    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subOutlierCloud;
    ros::Subscriber subImu;

    ros::Publisher pubCornerPointsSharp;
    ros::Publisher pubCornerPointsLessSharp;
    ros::Publisher pubSurfPointsFlat;
    ros::Publisher pubSurfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;

    pcl::VoxelGrid<PointType> downSizeFilter;

    double timeScanCur;
    double timeNewSegmentedCloud;
    double timeNewSegmentedCloudInfo;
    double timeNewOutlierCloud;

    bool newSegmentedCloud;
    bool newSegmentedCloudInfo;
    bool newOutlierCloud;

    cloud_msgs::cloud_info segInfo;
    std_msgs::Header cloudHeader;

    int systemInitCount;
    bool systemInited;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    int imuPointerFront;
    int imuPointerLast;
    int imuPointerLastIteration;

    float imuRollStart, imuPitchStart, imuYawStart;
    float cosImuRollStart, cosImuPitchStart, cosImuYawStart, sinImuRollStart, sinImuPitchStart, sinImuYawStart;
    float imuRollCur, imuPitchCur, imuYawCur;

    float imuVeloXStart, imuVeloYStart, imuVeloZStart;
    float imuShiftXStart, imuShiftYStart, imuShiftZStart;

    float imuVeloXCur, imuVeloYCur, imuVeloZCur;
    float imuShiftXCur, imuShiftYCur, imuShiftZCur;

    float imuShiftFromStartXCur, imuShiftFromStartYCur, imuShiftFromStartZCur;
    float imuVeloFromStartXCur, imuVeloFromStartYCur, imuVeloFromStartZCur;

    float imuAngularRotationXCur, imuAngularRotationYCur, imuAngularRotationZCur;
    float imuAngularRotationXLast, imuAngularRotationYLast, imuAngularRotationZLast;
    float imuAngularFromStartX, imuAngularFromStartY, imuAngularFromStartZ;

    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];
    float imuYaw[imuQueLength];

    float imuAccX[imuQueLength];
    float imuAccY[imuQueLength];
    float imuAccZ[imuQueLength];

    float imuVeloX[imuQueLength];
    float imuVeloY[imuQueLength];
    float imuVeloZ[imuQueLength];

    float imuShiftX[imuQueLength];
    float imuShiftY[imuQueLength];
    float imuShiftZ[imuQueLength];

    float imuAngularVeloX[imuQueLength];
    float imuAngularVeloY[imuQueLength];
    float imuAngularVeloZ[imuQueLength];

    float imuAngularRotationX[imuQueLength];
    float imuAngularRotationY[imuQueLength];
    float imuAngularRotationZ[imuQueLength];



    ros::Publisher pubLaserCloudCornerLast;
    ros::Publisher pubLaserCloudSurfLast;
    ros::Publisher pubLaserOdometry;
    ros::Publisher pubOutlierCloudLast;

    int skipFrameNum;
    bool systemInitedLM;

    int laserCloudCornerLastNum;
    int laserCloudSurfLastNum;

    int *pointSelCornerInd;
    float *pointSearchCornerInd1;
    float *pointSearchCornerInd2;

    int *pointSelSurfInd;
    float *pointSearchSurfInd1;
    float *pointSearchSurfInd2;
    float *pointSearchSurfInd3;

    float transformCur[6];
    float transformSum[6];

    float imuRollLast, imuPitchLast, imuYawLast;
    float imuShiftFromStartX, imuShiftFromStartY, imuShiftFromStartZ;
    float imuVeloFromStartX, imuVeloFromStartY, imuVeloFromStartZ; 

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;

    nav_msgs::Odometry laserOdometry;

    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform laserOdometryTrans;

    bool isDegenerate;
    cv::Mat matP;

    int frameCount;

public:

    FeatureAssociation():               //220309 构造函数，定义各种ros信息交流话题
        nh("~")
        {   
            //220403
            //订阅话题：   名称                                 类型                         回调函数    
            //  "/segmented_cloud"                      sensor_msgs::PointCloud2        laserCloudHandler
            //  "/segmented_cloud_info"                 cloud_msgs::cloud_info          laserCloudInfoHandler
            //  "/outlier_cloud"                        sensor_msgs::PointCloud2        outlierCloudHandler    
            //  imuTopic("/imu/data"在utility.h传入)     sensor_msgs::Imu                   imuHandler
            
            //  通过指令可查看ros自带的消息类型具体结构，如：rosmsg show sensor_msgs/Imu

            //220405can 接收来自imageProjection.cpp分割后的点云，也就是接受其发出来的话题
            //segmented_cloud 分割好的点云
            //segmented_cloud_info 分割好的点云，和segmented_cloud的区别是，cloud_info的强度信息是距离，而segmented_cloud的是range image的行列信息
            //outlier_cloud 离群点，也就是分割失败的点
            // 220320can 订阅和发布各种话题
            subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 1, &FeatureAssociation::laserCloudHandler, this);
            subLaserCloudInfo = nh.subscribe<cloud_msgs::cloud_info>("/segmented_cloud_info", 1, &FeatureAssociation::laserCloudInfoHandler, this);
            subOutlierCloud = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud", 1, &FeatureAssociation::outlierCloudHandler, this);
            subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 50, &FeatureAssociation::imuHandler, this);

            //220403
            //发布话题：   名称                                     类型                        用途          
            //  "/laser_cloud_sharp"                    sensor_msgs::PointCloud2    
            //  "/laser_cloud_less_sharp"               sensor_msgs::PointCloud2
            //  "laser_cloud_flat"                      sensor_msgs::PointCloud2
            //  "laser_cloud_less_flat"                 sensor_msgs::PointCloud2
            //  "/laser_cloud_corner_last"              sensor_msgs::PointCloud2        发布最新
            //  "/laser_cloud_surf_last"                sensor_msgs::PointCloud2        发布最新面surf点云
            //  "/outlier_cloud_last"                   sensor_msgs::PointCloud2        发布　离群　点云
            //  "/laser_odom_to_init"                   sensor_msgs::PointCloud2        发布激光odom数据
            
            //220405 发布点云
            pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
            pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
            pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("laser_cloud_flat", 1);
            pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("laser_cloud_less_flat", 1);
            
            pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last",2);//220308 为啥这里设置为２，上面都是１？
            pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
            pubOutlierCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
            pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 5);

            initializationValue();
        }

    void initializationValue()  //220309 初始化各个变量
    {   //220309 开辟一堆内存，存点的，不过感觉有点疑惑，这些都是单帧的最大点数量，这么做是为了保证肯定足够大吗？
        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];

        pointSelCornerInd = new int[N_SCAN*Horizon_SCAN];
        pointSearchCornerInd1 = new float[N_SCAN*Horizon_SCAN];
        pointSearchCornerInd2 = new float[N_SCAN*Horizon_SCAN];

        pointSelSurfInd = new int[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd1 = new float[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd2 = new float[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd3 = new float[N_SCAN*Horizon_SCAN];

        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);  //220309 这里下采样的参数含义是啥？　如下
                                    // 220320can 下采样滤波器设置叶子间距，就是格子就是最小距离（个人理解间隔多远取一个点）
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

        surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

        timeScanCur = 0;
        timeNewSegmentedCloud = 0;
        timeNewSegmentedCloudInfo = 0;
        timeNewOutlierCloud = 0;

        newSegmentedCloud = false;
        newSegmentedCloudInfo = false;
        newOutlierCloud = false;

        systemInitCount = 0;
        systemInited = false;

        imuPointerFront = 0;
        imuPointerLast = -1;    //220309 这里取　－１理由是什么？
        imuPointerLastIteration = 0;

        imuRollStart = 0; imuPitchStart = 0; imuYawStart = 0;
        cosImuRollStart = 0; cosImuPitchStart = 0; cosImuYawStart = 0;
        sinImuRollStart = 0; sinImuPitchStart = 0; sinImuYawStart = 0;
        imuRollCur = 0; imuPitchCur = 0; imuYawCur = 0;

        imuVeloXStart = 0; imuVeloYStart = 0; imuVeloZStart = 0;
        imuShiftXStart = 0; imuShiftYStart = 0; imuShiftZStart = 0;

        imuVeloXCur = 0; imuVeloYCur = 0; imuVeloZCur = 0;
        imuShiftXCur = 0; imuShiftYCur = 0; imuShiftZCur = 0;

        imuShiftFromStartXCur = 0; imuShiftFromStartYCur = 0; imuShiftFromStartZCur = 0;
        imuVeloFromStartXCur = 0; imuVeloFromStartYCur = 0; imuVeloFromStartZCur = 0;

        imuAngularRotationXCur = 0; imuAngularRotationYCur = 0; imuAngularRotationZCur = 0;
        imuAngularRotationXLast = 0; imuAngularRotationYLast = 0; imuAngularRotationZLast = 0;
        imuAngularFromStartX = 0; imuAngularFromStartY = 0; imuAngularFromStartZ = 0;

        for(int i = 0; i < imuQueLength; ++i)
        {
            imuTime[i] = 0;
            imuRoll[i] = 0; imuPitch[i] = 0; imuYaw[i] = 0;
            imuAccX[i] = 0; imuAccY[i] = 0; imuAccZ[i] = 0;
            imuVeloX[i] = 0; imuVeloY[i] = 0; imuVeloZ[i] = 0;
            imuShiftX[i] = 0; imuShiftY[i] = 0; imuShiftZ[i] = 0;
            imuAngularVeloX[i] = 0; imuAngularVeloY[i] = 0; imuAngularVeloZ[i] = 0;
            imuAngularRotationX[i] = 0; imuAngularRotationY[i] = 0; imuAngularRotationZ[i] = 0;
        }


        skipFrameNum = 1;

        for(int i = 0; i < 6; ++i){
            transformCur[i] = 0;
            transformSum[i] = 0;
        }

        systemInitedLM = false;

        imuRollLast = 0; imuPitchLast = 0; imuYawLast = 0;
        imuShiftFromStartX = 0; imuShiftFromStartY = 0; imuShiftFromStartZ = 0;
        imuVeloFromStartX = 0; imuVeloFromStartY = 0; imuVeloFromStartZ = 0;

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());

        laserOdometry.header.frame_id = "/camera_init";
        laserOdometry.child_frame_id = "/laser_odom";
        
        laserOdometryTrans.frame_id_ = "/camera_init";
        laserOdometryTrans.child_frame_id_ = "/laser_odom";

        isDegenerate = false;
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        frameCount = skipFrameNum;
    }

    void updateImuRollPitchYawStartSinCos(){    //2200309 更新imu的cos,sin值
        cosImuRollStart = cos(imuRollStart);    //220320can 更新初始时刻 i = 0时，rpy的sin cos值
        cosImuPitchStart = cos(imuPitchStart);
        cosImuYawStart = cos(imuYawStart);
        sinImuRollStart = sin(imuRollStart);
        sinImuPitchStart = sin(imuPitchStart);
        sinImuYawStart = sin(imuYawStart);
    }

    //220404can 计算局部坐标系下点云中的点相对第一个开始点的由于加减速运动产生的位移畸变
    void ShiftToStartIMU(float pointTime)   //220320 转换到imu的start坐标系
    {  //220309 切换到imu初始状态？？下面的数学原理不懂．220320can 三个量表示的是世界坐标系下，从start到cur的坐标的漂移量
        //220404can 计算相对于第一个点由于加减速产生的畸变位移(漂移量s = sCur - sStart - vt)
        imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
        imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
        imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloYStart * pointTime;
        //220320can 从世界坐标系变换到start坐标系（这里的start坐标系指的是，样本点云的最初那一帧位置吗？是的，在队列中的第一个）
        //220404 绕y轴旋转
        float x1 = cosImuYawStart *imuShiftFromStartXCur - sinImuYawStart * imuShiftFromStartZCur;
        float y1 = imuShiftFromStartYCur;
        float z1 = sinImuYawStart * imuShiftFromStartXCur + cosImuYawStart * imuShiftFromStartZCur;
        //220404 绕x旋转
        float x2 = x1;
        float y2 = cosImuPitchStart * y1 + sinImuPitchStart * z1;
        float z2 = -sinImuPitchStart * y1 + cosImuPitchStart * z1;
        //220404 绕ｚ旋转
        imuShiftFromStartXCur = cosImuRollStart * x2 + sinImuRollStart * y2;
        imuShiftFromStartYCur = -sinImuRollStart * x2 + cosImuRollStart * y2;
        imuShiftFromStartZCur = z2;
    }

    void VeloToStartIMU()   //220309 这里velo具体指的是什么，什么转换到imu上？见下一行注释
    {
        //220320can imuVeloXStart, imuVeloYStart, imuVeloZStart是点云索引　i = 0 时刻的速度
        //220320can 此处计算的是相对于初始时刻i = 0 时的相对速度
        //220320can 这个相对速度在世界坐标系下定义的
        imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
        imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
        imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;
        //220320can
        //  下面从世界坐标系转换到start坐标系，roll,pitch,yaw要取负值
        //  首先绕ｙ轴进行旋转
        //      |cosry  0 sinry|
        // Ry = |0      1     0|
        //      |-sinry 0 cosry|
        //220320can
        float x1 = cosImuYawStart * imuVeloFromStartXCur - sinImuYawStart * imuVeloFromStartZCur;
        float y1 = imuVeloFromStartYCur;
        float z1 = sinImuYawStart * imuVeloFromStartXCur + cosImuYawStart * imuVeloFromStartZCur;
        //220320can
        //  绕当前ｘ轴旋转（-pitch）的角度
        //      | 1    0          0|
        // Rx=  | 0   cosrx  -sinrx|
        //      | 0   sinrx   cosrx|
        //220320can
        float x2 = x1;
        float y2 = cosImuPitchStart * y1 + sinImuPitchStart * z1;
        float z2 = -sinImuPitchStart * y1 + cosImuPitchStart * z1;
        //220320can
        //绕当前ｚ轴旋转（-roll）角度
        //       | cosrz  -sinrz  0|
        // Rz  = | sinrz  cosrz   0|
        //       |   0      0     1|
        //220320can
        imuVeloFromStartXCur = cosImuRollStart * x2 + sinImuRollStart * y2;
        imuVeloFromStartYCur = -sinImuRollStart * x2 + cosImuRollStart * y2;
        imuVeloFromStartZCur = z2;
    }

    void TransformToStartIMU(PointType *p)  //220309 变换到开始的imu？ 220320can 该函数的功能是把点云坐标系变换到初始时刻的imu坐标系下
    {   //220309 下面的数学原理不懂
        //220320can
        //  因为在adjustDistortion函数中有对xyz的坐标进行交换的过程
        //  交换过程是 x = 原来的ｙ，　ｙ＝原来的ｚ，　ｚ＝原来的ｘ
        //  所以下面其实是绕ｚ轴（原来的ｘ轴）旋转，对应的roll角
        //       |cosrz  -sinrz  0|
        //  Rz = |sinrz  cosrz   0|
        //       |0       0      1|
        //[x1, y1, z1] ^ T = Rz * [x, y, z] 
        //
        // 因为imuHandler中进行过坐标变换
        // 所以下面的roll其实已经对应于新坐标系中(X-Y-Z)的yaw
        //220320can
        float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
        float y1 = sin(imuRollCur) * p->y + cos(imuRollCur) * p->y;
        float z1 = p->z;
        //220320can
        // 绕Ｘ轴（原来的ｙ轴）旋转
        //
        // [x2,y2,z2]^T=Rx*[x1,y1,z1]
        //    |1     0        0|
        // Rx=|0   cosrx -sinrx|
        //    |0   sinrx  cosrx|
        //220320can
        float x2 = x1;
        float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
        float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;
        //220320can
        // 最后再绕Y轴(原先的Z轴)旋转
        //    |cosry   0   sinry|
        // Ry=|0       1       0|
        //    |-sinry  0   cosry|
        //220320can
        float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
        float y3 = y2;
        float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;
        //220320can
        //下面部分代码功能是从imu坐标的原点变换到i = 0时imu初始时刻（从世界坐标系变换到start坐标系）（指的是从imu世界坐标系变换到imu开始时刻的坐标系吗？）
        //变换方式和函数VeloToStartIMU()中的类似
        //变换顺序：　Cur---> 世界坐标系　---> Start ,两次变换中    （Cur，Start两个坐标系，cur表示当前时刻，start表示的是开始时刻）
        //前一次是正变换，角度为正，后一次是逆变换，角度应该为负
        //参考：https://blog.csdn.net/wykxwyc/article/details/101712524
        //220320can
        float x4 = cosImuYawStart * x3 - sinImuYawStart * z3;
        float y4 = y3;
        float z4 = sinImuYawStart * x3 + cosImuYawStart * z3;

        float x5 = x4;
        float y5 = cosImuPitchStart * y4 + sinImuPitchStart * z4;
        float z5 = -sinImuPitchStart * y4 +cosImuPitchStart * z4;
        //220320can
        //绕ｚ轴（原来的ｘ轴）变换角度到初始imu时刻，另外需要加上imu的位移漂移
        //后面加上的　imuShiftFromStart..表示从start时刻到cur时刻的漂移，
        // (imuShiftFromStart..在start坐标系下)
        //220320can
        p->x = cosImuRollStart * x5 + sinImuRollStart * y5 + imuShiftFromStartXCur;
        p->y = -sinImuRollStart * x5 + cosImuRollStart * y5 + imuShiftFromStartYCur;
        p->z = z5 + imuShiftFromStartZCur;
    }
    //220403 对三者进行积分，获取每帧imu数据在全局坐标系下的位移、速度、角旋转量
    void AccumulateIMUShiftAndRotation()    //220309  移动转动imu积分？220320 位移，速度，角速度积分
    {   //220309 下面的数学原理是什么？
        //220403can 来源：https://blog.csdn.net/liuyanpeng12333/article/details/82737181
        //获得由imuHandle函数得到该帧IMU数据的欧拉角和三轴角加速度
        float roll = imuRoll[imuPointerLast];
        float pitch = imuPitch[imuPointerLast];
        float yaw = imuYaw[imuPointerLast];
        float accX = imuAccX[imuPointerLast];
        float accY = imuAccY[imuPointerLast];
        float accZ = imuAccZ[imuPointerLast];
        //220320can
        //先绕Ｚ轴（原来的ｘ轴）旋转，下方坐标系表示imuHandler()中加速度的坐标变换
        //  z->Y
        //  ^  
        //  |    ^ y->X
        //  |   /
        //  |  /
        //  | /
        //  -----> x->Z
        //
        //     |cosrz  -sinrz  0|
        //  Rz=|sinrz  cosrz   0|
        //     |0       0      1|
        //　［x1, y1, z1]^T = Rz*[accX, accY, accZ]
        //  因为在imuHandler中进行过坐标转换
        //  所以下面的roll其实已经对应于新的坐标系中（X-Y-Z）的yaw
        //220320can
        //220403can 将当前时刻的加速度值绕交换过的ZXY固定轴（原XYZ）分别旋转
        //220403can (roll, pitch, yaw)角，转换得到世界坐标系下的加速度值(右手法则)
        //220403can 绕z轴旋转(roll，对应的是原来的x轴)
        float x1 = cos(roll) * accX - sin(roll) * accY;
        float y1 = sin(roll) * accX + cos(roll) * accY;
        float z1 = accZ;
        //220320can
        // 绕X轴(原y轴，所以是pitch)旋转
        // [x2,y2,z2]^T=Rx*[x1,y1,z1]
        //    |1     0        0|
        // Rx=|0   cosrx -sinrx|
        //    |0   sinrx  cosrx|
        //220320can
        //220403can 绕x轴旋转(pitch)
        float x2 = x1;
        float y2 = cos(pitch) * y1 - sin(pitch) * z1;
        float z2 = sin(pitch) * y1 + cos(pitch) * z1;
        //220320can
        // 最后再绕Y轴(原z轴,所以下面是yaw)旋转
        //    |cosry   0   sinry|
        // Ry=|0       1       0|
        //    |-sinry  0   cosry|
        //220320can
        //220403can 绕y轴旋转(yaw)
        accX = cos(yaw) * x2 + sin(yaw) * z2;
        accY = y2;
        accZ = -sin(yaw) * x2 + cos(yaw) * z2;
        //220320can 进行位移，速度，角旋转量的累加。也就是imu的积分操作
        //220403can 上一个imu点
        int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;    //220309 这个变量表示啥含义？ 220403 表示上一个imu点
        //220403can 上一个点到当前点所经历的时间，即计算imu测量周期
        double timeDiff  = imuTime[imuPointerLast] - imuTime[imuPointerBack];
        //220403can 要求imu的频率至少比lidar高，这样的imu信息才使用，后面校正也才有意义
        if(timeDiff < scanPeriod){  //220309　时间间隔小于单圈扫描时间则进行下面计算，那么问题来了，下来的转换数学原理是啥？ 220403这里的时间含义是，要求imu的频率比lidar的频率高才行使用意思
            //220403can
            //隐含从静止开始运动
            //求每个imu时间点的位移、速度、角旋转量
            //220403can
            //220320 求位移
            imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff +accX * timeDiff * timeDiff /2; //220320 后面可视为at^2/2,如同一个加速度求位移积累公式
            imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff +accY * timeDiff * timeDiff /2;
            imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff +accZ * timeDiff * timeDiff /2;
            //220320 求速度
            imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
            imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
            imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
            //220320 求角旋转量
            imuAngularRotationX[imuPointerLast] = imuAngularRotationX[imuPointerBack] + imuAngularVeloX[imuPointerBack] * timeDiff;
            imuAngularRotationY[imuPointerLast] = imuAngularRotationY[imuPointerBack] + imuAngularVeloY[imuPointerBack] * timeDiff;
            imuAngularRotationZ[imuPointerLast] = imuAngularRotationZ[imuPointerBack] + imuAngularVeloZ[imuPointerBack] * timeDiff;
        }
    }

    //220403can 接收imu消息，imu坐标系为x轴向前，y轴向右，z轴向上的右手坐标系。该函数的作用是运动补偿
    
    //220403can 来源：https://blog.csdn.net/liuyanpeng12333/article/details/82737181
    //我们知道IMU的数据可以提供给我们IMU坐标系三个轴相对于世界坐标系的欧拉角和三个轴上的加速度。
    //但由于加速度受到重力的影响所以首先得去除重力影响。在去除重力影响后我们想要获得IMU在世界坐
    //标系下的运动，因此根据欧拉角就可以将IMU三轴上的加速度转换到世界坐标系下的加速度。 然后根
    //据加速度利用 公式s1 = s2+ vt + at*t*1/2来计算位移。
    //因此我们可以求出每一帧IMU数据在世界坐标系下对应的位移和速度。

    //220405can 接收imu消息然后保存到数组
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)    //220309 imu头信息？ 220403imu数据流的回调函数
    {
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation); //220321 传感器四元数信息转为TF四元数信息
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);    //220321 从tf四元数信息获取欧拉角
        //220321can 加速度去除重力影响，同时进行坐标轴变换（原始的坐标系,XYZ---RPY）
        //220403can 减去重力的影响,求出xyz方向的加速度实际值，并进行坐标轴交换，统一到z轴向前,
        //220403can x轴向左的右手坐标系, 交换过后RPY对应fixed axes ZXY(RPY---ZXY)。
        //220403can Now R = Ry(yaw)*Rx(pitch)*Rz(roll)
        float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;  //220310 9.81是重力加速度吗？
        float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
        float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;  //220310 这里几条的数学含义是啥？去除重力影响，并进行坐标轴名称变换（XYZ-ZXY）,注意轴的欧拉角是不变的
        //220403can  循环移位效果，形成环形数组
        imuPointerLast = (imuPointerLast + 1) % imuQueLength;   //220310 这里的＋１的作用是什么？最后一帧ｉｍｕ数据加１？220403 目的形成环形数组

        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();  //220310 根据imu数据获取时间戳

        imuRoll[imuPointerLast] = roll;     //220310 最新imu帧的欧拉角
        imuPitch[imuPointerLast] = pitch;
        imuYaw[imuPointerLast] = yaw;

        imuAccX[imuPointerLast] = accX;     //220310 加速度？220404 是的
        imuAccY[imuPointerLast] = accY;
        imuAccZ[imuPointerLast] = accZ;

        imuAngularVeloX[imuPointerLast] = imuIn->angular_velocity.x; // 220310 角速度？220404是的
        imuAngularVeloY[imuPointerLast] = imuIn->angular_velocity.y;
        imuAngularVeloZ[imuPointerLast] = imuIn->angular_velocity.z;

        AccumulateIMUShiftAndRotation();
    }

    //220405can 接收"/segmented_cloud"消息，保存到segmentedCloud
    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){  //2203010 扫描点云的头信息？

        cloudHeader = laserCloudMsg->header;

        timeScanCur = cloudHeader.stamp.toSec();    // 220310 获取分割点云的时间戳？使得
        timeNewSegmentedCloud = timeScanCur;

        segmentedCloud->clear();    //220310 清理分割点云，用于存储下一帧点云信息
        pcl::fromROSMsg(*laserCloudMsg, *segmentedCloud);   //220310 将ｒｏｓ的点云信息转为ｐｃｌ的点云信息

        newSegmentedCloud = true;
    }

    //220405can 接收"/outlier_cloud"消息，保存到outlierCloud
    void outlierCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){    //220310 外界点云头信息

        timeNewOutlierCloud = msgIn->header.stamp.toSec();

        outlierCloud->clear();      //220310 容器清空
        pcl::fromROSMsg(*msgIn, *outlierCloud);     //220310 从ros类型的点云转为pcl类型
        
        newOutlierCloud = true;
    }
    //220405can 接收"/segmented_cloud_info"消息，并且保存到segInfo中
    //220405can 这里没有转换为PCL点云，而是直接保存的消息cloud_msgs::cloud_info segInfo
    void laserCloudInfoHandler(const cloud_msgs::cloud_infoConstPtr& msgIn) //220310 扫射点云信息头消息
    {
        timeNewSegmentedCloudInfo = msgIn->header.stamp.toSec();
        segInfo = *msgIn;
        newSegmentedCloudInfo = true;
    }

    void adjustDistortion() //220310 降低失真？什么造成的失真？
    {    
        bool halfPassed = false;
        int cloudSize = segmentedCloud->points.size();

        PointType point;

        for(int i = 0; i < cloudSize; i++){
            //220323can 这里的xyz与laboshin_loam代码中的一样经过坐标轴变换（这里的laboshin_loam指的是什么？）
            //20323can imuhandler()中相同的坐标变换
            point.x = segmentedCloud->points[i].y;
            point.y = segmentedCloud->points[i].z;
            point.z = segmentedCloud->points[i].x;

            //220323can 
            //-atan2(p.x, p.z) ==> -atan2(y, x)
            //ori表示的是偏航角yaw,因为前面有负号，ori = [-M_PI, M_PI]
            //因为segInfo.orientationDiff表示的范围是(PI, 3PI),在２PI附近
            //下面过程的主要作用就是调整ori大小，满足start<ori<end
            //220323can
            float ori = -atan2(point.x, point.z);
            if(!halfPassed){    //220310 这里面的数学原理不懂
                if (ori < segInfo.startOrientation - M_PI / 2)  //220310 开始方位角计算？
                        //220323can start-ori > M_PI/2,说明ori小于start，不合理
                        //220323can 正常情况在前半圈的话，ori-start范围［０,M_PI］(正常范围依据是什么？)
                    ori += 2 * M_PI;
                else if (ori > segInfo.startOrientation + M_PI * 3 / 2)
                        // 220323can ori-start > M_PI*3/2,说明ori太大，不合理
                    ori -= 2 * M_PI;
                
                if(ori - segInfo.startOrientation > M_PI)
                    halfPassed = true;
            }else{
                ori += 2 * M_PI;

                if(ori < segInfo.endOrientation - M_PI * 3 / 2) //220310 结束方位角计算？
                        //220323can end - ori > M_PI * 3/2,ori太小，不合理，（不合理就调整）
                    ori += 2 * M_PI;
                else if ( ori > segInfo.endOrientation + M_PI / 2)
                        //220323can  ori - end > M_PI/2,太大，不合理（不合理就调整）
                    ori -= 2 * M_PI;                      
            }

            //220323can 用point.intensity 来保存时间
            float relTime = (ori - segInfo.startOrientation) / segInfo.orientationDiff; //220310 这里作用是什么？什么的真实时间？
            point.intensity = int(segmentedCloud->points[i].intensity) + scanPeriod * relTime;  //220310 点的强度值根据现有的分割点分辨率和时间计算获取？

            if(imuPointerLast >= 0){    // 220310 这里的含义是什么？
                float pointTime = relTime * scanPeriod; //220324 reltime可以理解为第几次，后面是周期时间，从而得到点云的时间？
                imuPointerFront = imuPointerLastIteration;
                //220323can while 循环内进行时间轴对齐
                while(imuPointerFront != imuPointerLast){
                    if(timeScanCur + pointTime < imuTime[imuPointerFront]){     //220310 这里的时间条件含义是什么？
                        break;
                    }
                    imuPointerFront = (imuPointerFront + 1) % imuQueLength;     //220310 这里的＋１的原理是什么？最后结果作用是什么？
                }

                if(timeScanCur + pointTime > imuTime[imuPointerFront]){ //220310 这里的判断条件处于什么考虑？
                    //220323can 该条件内imu数据比激光数据早，但是没有更新后面的数据
                    //220323can (打个比方，激光在９点，imu现在只有８点的)
                    //220323can 这种情况上面while循环是以imuPointerFront == imuPointerLast结束的
                    imuRollCur = imuRoll[imuPointerFront];
                    imuPitchCur = imuPitch[imuPointerFront];
                    imuYawCur = imuYaw[imuPointerFront];

                    imuVeloXCur = imuVeloX[imuPointerFront];
                    imuVeloYCur = imuVeloY[imuPointerFront];
                    imuVeloZCur = imuVeloZ[imuPointerFront];    //220310 这些变量含义是什么？

                    imuShiftXCur = imuShiftX[imuPointerFront];
                    imuShiftYCur = imuShiftY[imuPointerFront];
                    imuShiftZCur = imuShiftZ[imuPointerFront];
                }else{  //220310 下面这些变量含义是什么？背后的数学公式是什么？
                    //220324can
                    //在imu数据充足情况下可以进行插补
                    //当前timeScanCur + pointTime < imuTime[imuPointerFront]
                    //而且imuPointerFront是最早的一个时间大于timeScanCur + pointTime 的imu数据指针
                    //220324can
                    int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;   //220310 变量含义？
                    float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack])
                                                    / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                    float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime)
                                                    / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                    
                    //220324can
                    //通过上面计算的ratioFront以及ratioBack进行插补
                    //因为imuRollCur和imuPitchCur通常都在０度左右，变化不会很大，因此不需要考虑超过２M_PI情况
                    //imuYaw转的角度比较大，需要考虑超过２*M_PI的情况(这里的角度范围依据是什么？)
                    //220324can
                    imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
                    imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
                    if(imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI){
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
                    }else if(imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < - M_PI){
                            imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
                    }else{
                            imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] * ratioBack);
                    }//220310 这几个公式看的我眼花了．．．数学原理是什么呢？插补
                    //220324can imu速度插补
                    imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
                    imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
                    imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;
                    //220324can imu位置插补
                    imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
                    imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
                    imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;     //220310 问题同上，数学原理是什么？
                }

                if(i == 0){     //220310 这里的０依据是什么？
                    //220324can 此处更新过的角度值主要用在updateImuRollPitchYawStartSinCos()中，
                    //220324can 更新每个角的正余炫
                    imuRollStart = imuRollCur;
                    imuPitchStart = imuPitchCur;
                    imuYawStart = imuYawCur;

                    imuVeloXStart = imuVeloXCur;
                    imuVeloYStart = imuVeloYCur;
                    imuVeloZStart = imuVeloZCur;

                    imuShiftXStart = imuShiftXCur;
                    imuShiftYStart = imuShiftYCur;
                    imuShiftZStart = imuShiftZCur;

                    if(timeScanCur + pointTime > imuTime[imuPointerFront]){
                        //220324can 这条件内imu数据比激光数据早，但是没有更后面的数据
                        imuAngularRotationXCur = imuAngularRotationX[imuPointerFront];
                        imuAngularRotationYCur = imuAngularRotationY[imuPointerFront];
                        imuAngularRotationZCur = imuAngularRotationZ[imuPointerFront];
                    }else{
                        //220324can 在imu数据充足的情况下可以进行插补
                        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack])
                                            / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime)
                                            / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                        imuAngularRotationXCur = imuAngularRotationX[imuPointerFront] * ratioFront + imuAngularRotationX[imuPointerBack] * ratioBack;
                        imuAngularRotationYCur = imuAngularRotationY[imuPointerFront] * ratioFront + imuAngularRotationY[imuPointerBack] * ratioBack;
                        imuAngularRotationZCur = imuAngularRotationZ[imuPointerFront] * ratioFront + imuAngularRotationZ[imuPointerBack] * ratioBack;
                    }// 220310 上面的数学原理以及变量含义是什么？插补操作
                    //220324can 距离上一次插补，旋转过的角度变化值
                    imuAngularFromStartX = imuAngularRotationXCur - imuAngularRotationXLast;
                    imuAngularFromStartY = imuAngularRotationYCur - imuAngularRotationYLast;
                    imuAngularFromStartZ = imuAngularRotationZCur - imuAngularRotationZLast;

                    imuAngularRotationXLast = imuAngularRotationXCur;
                    imuAngularRotationYLast = imuAngularRotationYCur;
                    imuAngularRotationZLast = imuAngularRotationZCur;

                    //220324can 这里更新的是i = 0时刻的rpy角，后面将速度坐标投影过来会用到i = 0 时刻的值
                    updateImuRollPitchYawStartSinCos();     //220310　更新ｒｐｙ，sin,cos值？
                }else{
                    //220324can 速度投影到初始i = 0时刻
                    VeloToStartIMU();       //220310 Velo是什么含义？速度，单词的部分
                    TransformToStartIMU(&point);    //220310 点信息转换到imu
                }
            }
            segmentedCloud->points[i] = point;      //220310 将点保存到分割点云中？那这个点是怎么来的？
        }

        imuPointerLastIteration = imuPointerLast;   //220310 者里两个变量的含是什么？
    }
    //220324can 计算光滑性，这里的计算没有完全按照公式进行,（这里的公式指的是？论文中有，但是代码不完全和论文公式相同）
    //220324can 却还扫除以总点数i和r[i]
    //220405can 平滑度的计算是取当前点的左边5个点和右边5个点和当前点的深度差值，然后求平方
    //而论文中的计算方法和代码中的不一样，论文中是取深度差的平均值，然后除以自身的模，也就是说论文
    //中的曲率和尺度无关，不管当前点离lidar近还是远，只要曲率超过一定的值就可以，同时在仿真环境也
    //可以解决尺度的问题，而代码中直接取得是绝对大小。(https://zhuanlan.zhihu.com/p/384902839)
    void caluclateSmoothness()  //220311 计算平滑度？220405是的
    {
        int cloudSize = segmentedCloud->points.size();  //220311 点云的数量
        for(int i = 5; i < cloudSize - 5; i++){
            //220405can 计算相邻的10个点深度差的和
            float diffRange = segInfo.segmentedCloudRange[i-5] + segInfo.segmentedCloudRange[i-4]
                            + segInfo.segmentedCloudRange[i-3] + segInfo.segmentedCloudRange[i-2]
                            + segInfo.segmentedCloudRange[i-1] + segInfo.segmentedCloudRange[i] * 10
                            + segInfo.segmentedCloudRange[i+1] + segInfo.segmentedCloudRange[i+2]
                            + segInfo.segmentedCloudRange[i+3] + segInfo.segmentedCloudRange[i+4]
                            + segInfo.segmentedCloudRange[i+5]; //220311 这个公式计算的内容是什么？物理意义是什么？平滑度的衡量指标？
            //220405can 取平方
            cloudCurvature[i] = diffRange*diffRange;

            //220326can 在markOccludedPoints()函数中对该函数进行重新修改
            cloudNeighborPicked[i] = 0;
                            //220326can 在extractFeatures()函数中对标签进行修改，
                            //220326can 初始化为０，　surfPointsFlat标记为－１，surfPointsLessFlatScan为不大于０的标签
                            //220326can cornerPointsSharp标记为２，cornerPointsLessSharp标记为１
            cloudLabel[i] = 0;
            //220405can 保存曲率，保存索引
            cloudSmoothness[i].value = cloudCurvature[i];   // 220311 保存平滑度数值？指的是点云曲率？
            cloudSmoothness[i].ind = i;                     // 220311 记录该点的ｉｄ？
        }
    }

    //220326can 阻塞点是哪种点？
    //220326can 阻塞点指的是点云之间相互遮挡，而且又靠得很近的点
    void markOccludedPoints()   //2200311 标记Occlouded点云，但是，这个是啥点呢？英文单词意思，阻塞点
    {
        int cloudSize = segmentedCloud->points.size();

        for(int i = 5; i < cloudSize -6; ++i){  //220311 开始和结束的５个点，不参与计算

            float depth1 = segInfo.segmentedCloudRange[i];
            float depth2 = segInfo.segmentedCloudRange[i+1];
            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[i+1] - segInfo.segmentedCloudColInd[i]));
            
            // 220405can 标记有遮挡的点
            if(columnDiff < 10){    //220311 这里的１０依据是什么？还有这个变量物理含义是什么？220405表示点的左右５个点
                // 220326can 选择距离较远的那些点，并把它们标记为１
                if(depth1 - depth2 > 0.3){  //220311 判断条件依据？下面的操作原理是？标记距离远的点
                    cloudNeighborPicked[i-5] = 1;
                    cloudNeighborPicked[i-4] = 1;
                    cloudNeighborPicked[i-3] = 1;
                    cloudNeighborPicked[i-2] = 1;
                    cloudNeighborPicked[i-1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if(depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i+1] = 1;
                    cloudNeighborPicked[i+2] = 1;
                    cloudNeighborPicked[i+3] = 1;
                    cloudNeighborPicked[i+4] = 1;
                    cloudNeighborPicked[i+5] = 1;
                    cloudNeighborPicked[i+6] = 1;
                }
            }
            //220311 下面两个变量，计算的是相邻两点的距离？
            float diff1 = std::abs(float(segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i]));
            float diff2 = std::abs(float(segInfo.segmentedCloudRange[i+1] - segInfo.segmentedCloudRange[i]));
            //220326can 选择距离变化比较大的点（离群点），并将它们标记为１
            if ( diff1 > 0.002 * segInfo.segmentedCloudRange[i] && diff2 > 0.002 * segInfo.segmentedCloudRange[i])
                cloudNeighborPicked[i] = 1; //220311 判断条件依据是什么？这里复制１的缘由是什么？标记变化比较大的点
        }
    }

    //220405can https://zhuanlan.zhihu.com/p/384902839
    //220405can 提取的特征分为2类，一类是曲率比较大的线特征，一类是曲率比较小的面特征
    //提取的方法是把平面划分为6等分，也就是6个方向（LOAM中为前、后、左、右4个方向），
    //根据上述计算好的曲率进行排序，然后每个方向最多选择2个线特征和4个面特征，如果超过则跳过，
    //在下一个方向上继续选择。如果一个点已经被选择为特征点，那么它的相邻点会被标记，不允许被
    //选为特征了。
    void extractFeatures()  // 220311 提取特征
    {
        cornerPointsSharp->clear();
        cornerPointsLessSharp->clear();
        surfPointsFlat->clear();
        surfPointsLessFlat->clear();

        for( int i = 0; i < N_SCAN; i++){

            surfPointsLessFlatScan->clear();
            //220405can 分为6个方向，每个方向分别选择线特征和面特征
            for(int j = 0; j < 6; j++){
                //220326can sp和ep的含义是什么？startPointer,endPointer
                //220311 这个循环条件依据是什么，下面两个变量物理意义是什么？
                int sp = (segInfo.startRingIndex[i] * (6-j) + segInfo.endRingIndex[i] * j) / 6;
                int ep = (segInfo.startRingIndex[i] * (5-j) + segInfo.endRingIndex[i] * (j+1)) / 6 - 1;
                //220405can 根据曲率排序
                if( sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin()+ep, by_value());    //220311  什么变量的升序排序？ 平滑度点云？

                int largestPickedNum = 0;   //220311 最大的选取数量？
                //220405can 选择线特征，不为地面，segInfo.segmentedCloudGroundFlag[ind] == false
                for(int k = ep; k > sp; k--){
                    //220326can 每次ind的值就是等于k?有什么意义？
                    //220326can 因为上面对cloudSmoothness进行了一次从小到大的排序，所以ind不一定等于k了
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] > edgeThreshold &&
                        segInfo.segmentedCloudGroundFlag[ind] == false){    //220311 这里的判断条件是什么？这里写的依据是什么？

                        largestPickedNum++;
                        //220405can 选择最多2个线特征
                        if(largestPickedNum <= 2){  //220311 物理含义是什么？ 220405 如上一行解析
                            //220326can 论文中nFe = 2, cloudSmoothness已经按照从小到大的顺序排列， 
                            //220326can　所以这边只要选择最后两个放进队列即可
                            //220326can cornerPointsSharp标定为２   （这部分个人没有理解）
                            cloudLabel[ind] = 2;
                            cornerPointsSharp->push_back(segmentedCloud->points[ind]);
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        }else if (largestPickedNum <= 20) {  //220405can 平滑一些的线特征20个，用于mapping 
                                            //220326can 塞20个点到cornerPointLessSharp中去
                                            //220326can cornerPointsLessSharp标记为１ 
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        }else{//220405can 超过则退出
                            break;
                        }

                        //220405can 标记相邻点，防止特征点过于集中
                        cloudNeighborPicked[ind] = 1;   //220311 下面两个循环如何理解？变量物理意义是什么？如上一行解析
                        for (int l = 1; l <= 5; l++){
                            //220326can 从ind + l开始后面的５个点，每个点index之间的差值，
                            //220326can　确保columnDiff <= 10,然后标记为我们需要的点
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l -1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--){
                                    //220326can 从ind + l 开始前面五个点，计算差值然后标记
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l +1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                //220311 上下两个for大循环，基于什么考虑的，对比差异性再哪里？物理含义是什么？ 220405 上面的防止特征点过于集中
                int smallestPickedNum = 0;  //220311 最小选取量？
                //220405can 选择面特征，为地面，segInfo.segmentedCloudGroundFlag[ind] == true
                for ( int k = sp; k <= ep; k++){    //220311 surfThreshold物理意义是什么？面阈值？
                    int ind = cloudSmoothness[k].ind;
                    //220326can 平面点只从地面点中进行选择？为什么这样做？220405这里是进行面特征的提取
                    if ( cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] < surfThreshold && 
                        segInfo.segmentedCloudGroundFlag[ind] == true){

                        cloudLabel[ind] = -1;
                        surfPointsFlat->push_back(segmentedCloud->points[ind]); //220311 容器的物理意义是什么？

                        //220326can 论文中nFp = 4,将４个最平的平面点放入队列中
                        //220405can 选择最多4个面特征
                        smallestPickedNum++;
                        if (smallestPickedNum >= 4){    //220311 判断条件依据？
                            break;
                        }

                        //220405can 标记相邻点
                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++){
                                //220326can 从后往前开始标记
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if ( columnDiff > 10)
                                break;
                            
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--){

                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if ( columnDiff > 10)
                                break;
                            
                            cloudNeighborPicked[ind + l] = 1;    
                        }
                    }
                }
                //220405can 选择的是地面的面特征，和其它没被选择的点（除了地面的点，并且不是线特征）
                for (int k = sp; k <= ep; k++){ //220311 这个for循环作用是什么？
                    if(cloudLabel[k] <= 0){ //220311 点云标签值，会小于０？或者说，这里的物理意义是什么？
                        surfPointsLessFlatScan->push_back(segmentedCloud->points[k]);
                    }
                }
            }
            //220326can surfPointsFlatScan中有过多的点云，如果点云太多，计算量太大
            //220326can 进行下采样，可以大大减少计算量
            //220405can 下采样平滑一些的面特征
            surfPointsLessFlatScanDS->clear();  //220311 下采样常规操作，那么，基于什么考虑进行下采样操作的？
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.filter(*surfPointsLessFlatScanDS);

            *surfPointsLessFlat += *surfPointsLessFlatScanDS;
        }
    }
    //220405can  https://zhuanlan.zhihu.com/p/384902839
    //特征点的选择满足以下3个条件:
    //选择的边缘点或平面点的数量不能超过每个方向上的最大值，一共有6个方向，每个方向上最多2个线特征，4个面特征
    //线特征和面特征周围相邻的点不能被选中
    //不能是平行于激光雷达光束的点或者被遮挡的点

    //特征提取中，代码输出：
    // cornerPointsSharp 线特征（不为地面），每个方向上最多2个
    // cornerPointsLessSharp 比cornerPointsSharp平滑的线特征（不为地面），每个方向上20个
    // surfPointsFlat 面特征（为地面），每个方向上最多4个
    // surfPointsLessFlat 面特征（地面或者分割点云中不为线特征的点）

    void publishCloud() //220311 发布点云
    {
        sensor_msgs::PointCloud2 laserCloudOutMsg;
            //220311 判断有人订阅这个话题，则发布
        if (pubCornerPointsSharp.getNumSubscribers()!=0){
            pcl::toROSMsg(*cornerPointsSharp, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = cloudHeader.stamp;
            laserCloudOutMsg.header.frame_id = "/camera";
            pubCornerPointsSharp.publish(laserCloudOutMsg);
        }

        if (pubCornerPointsLessSharp.getNumSubscribers()!=0){
            pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = cloudHeader.stamp;
            laserCloudOutMsg.header.frame_id = "/camera";
            pubCornerPointsLessSharp.publish(laserCloudOutMsg);
        }

        if (pubSurfPointsFlat.getNumSubscribers()!=0){
            pcl::toROSMsg(*surfPointsFlat, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = cloudHeader.stamp;
            laserCloudOutMsg.header.frame_id = "/camera";
            pubSurfPointsFlat.publish(laserCloudOutMsg);
        }

        if (pubSurfPointsLessFlat.getNumSubscribers()!=0){
            pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = cloudHeader.stamp;
            laserCloudOutMsg.header.frame_id = "/camera";
            pubSurfPointsLessFlat.publish(laserCloudOutMsg);
        }
    }








































    //220404 对于TransformToStart() TransformToEnd(）配合作用，对于当前和之前两帧点云，之前的点云通过
    //TransformToEnd(）函数，都变换到其帧的结束时刻；当前帧的点云，通过TransformToStart()变换到其帧开始时刻
    //由于当前帧的开始时刻也是上一帧的结束时刻，两帧点云在时间上拼接起来了
    //220404can 这两个函数的作用就是，去除畸变，将两帧点云数据统一到同一个坐标系下计算

    //220403can 给每帧点云去畸变处理（匀速运动假设），最后得到的点云可以认为是在开始时刻静止扫描得到的
    void TransformToStart(PointType const * const pi, PointType * const po) //220312 点到开始的转换，pi指的是输入，po指的是输出（这里的start物理含义是什么？）
    {   //220403  https://www.cnblogs.com/ReedLW/p/9005621.html
        //在进行KDTree最近点搜索前，首先将进行畸变处理后的点云转换到每一次扫描的开始时刻
        //根据匀速运动假设（假设一帧扫描属于匀速运动，平移和旋转都匀速的过程）计算出当前点时刻Lidar的位移和旋转
        //
        //220327can
        // intensity代表的是：整数部分ring序号，小数部分是当前点在这一圈所花的时间
        //　关于intensity, 参考　adjustDistortion()函数中的定义
        //  s 代表的其实一个比例,s的计算方法应该如下：
        //　s = (pi->intensity - int(pi->intensity))/SCAN_PERIOD
        //  ===> SCAN_PERIOD = 0.1(雷达频率为１０hz)
        // https://github.com/laboshinl/loam_velodyne/issues/29
        //220327can
        //220403 线性插值系数，当前点扫描所花时间在单圈扫描时间（0.1秒）（扫描周期）的比例
        float s = 10 * (pi->intensity - int(pi->intensity));        //220403 插值系数
        //220403can 根据每个点在点云中的相对位置关系，乘以相应的旋转平移系数
        //220403 旋转角
        float rx = s * transformCur[0];
        float ry = s * transformCur[1];
        float rz = s * transformCur[2];
        //220403 平移距离
        float tx = s * transformCur[3];
        float ty = s * transformCur[4];
        float tz = s * transformCur[5];
        //220403 先平移后绕z旋转（-rz）
        float x1 = cos(rz) * (pi->x -tx) + sin(rz) * (pi->y -ty);
        float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y -ty);
        float z1 = (pi->z -tz);     //220312 变量的物理含义是什么？数学公式是什么？点坐标变换，绕ｚ轴旋转
        //220403can 绕x轴旋转（-rx）
        float x2 = x1;
        float y2 = cos(rx) * y1 + sin(rx) * z1;
        float z2 = -sin(rx) * y1 + cos(rx) * z1;    //220312 同上疑惑,变量物理意义以及数学公式？绕ｘ轴旋转
        //绕y轴旋转（-ry）
        po->x = cos(ry) * x2 - sin(ry) * z2;
        po->y = y2;
        po->z = sin(ry) * x2 + cos(ry) * z2;    //220327 完成绕ｙ轴的旋转，并且成功转移到输出点
        po->intensity = pi->intensity;      //220312 这里实现了输入与输出的点转换，但是背后的数学原理是什么？220403线性插值操作
    }

    //220329can 先转到start，再从start旋转到end     220403 给每帧点云去畸变处理
    //220403can 去除当前点云中的点相对第一个点因匀速运动（平移和旋转都是匀速的）产生的畸变，
    //220403can 最后获取的点云相当于在扫描结束位置静止扫描得到(https://blog.csdn.net/nh54zyt/article/details/116028175)
    void TransformToEnd(PointType const * const pi, PointType * const po)   //220312 点到结束的转换，pi指的是输入，po指的是输出（这里的end物理含义是什么？）
    {
        //220329can 关于ｓ,参考上面TransformToStart()的注释（个人不理解）。220403 线性插值的比例系数
        float s = 10 * (pi->intensity - int(pi->intensity));
        //220329 理解为坐标＋欧拉角，的比例成分。220403 线性插值的比例系数
        float rx = s * transformCur[0];
        float ry = s * transformCur[1];
        float rz = s * transformCur[2];
        float tx = s * transformCur[3];
        float ty = s * transformCur[4];
        float tz = s * transformCur[5];
        //220403 先平移，再绕z旋转
        //220403can 平移后绕z轴旋转（-rz）
        float x1 = cos(rz) * (pi->x -tx) + sin(rz) * (pi->y -ty);
        float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y -ty);
        float z1 = (pi->z -tz);     //220312 变量的物理含义是什么？数学公式是什么？220329 绕ｚ轴旋转
        //220403can 绕x轴旋转（-rx）
        float x2 = x1;
        float y2 = cos(rx) * y1 + sin(rx) * z1;
        float z2 = -sin(rx) * y1 + cos(rx) * z1;    //220312 同上疑惑,变量物理意义以及数学公式。220329 绕x轴旋转
        //220403can  绕y轴旋转（-ry）
        float x3 = cos(ry) * x2 - sin(ry) * z2;
        float y3 = y2;
        float z3 = sin(ry) * x2 +cos(ry) * z2;      //220312 这里跟TransformToStard的输出点转换一样，这里的物理意义是什么？数学原理？　220329 绕ｙ轴旋转
                                                    //220403can 求出了相对于起始点校正的坐标(https://codeantenna.com/a/txBSKStHaA)
        //220329 理解为坐标＋欧拉角
        rx = transformCur[0];
        ry = transformCur[1];
        rz = transformCur[2];
        tx = transformCur[3];
        ty = transformCur[4];
        tz = transformCur[5];       //220312 重新赋值的理由是什么？注意，这里没有参数ｓ的
        //220329 下面表示绕ｙｘｚ旋转
        //220403can　绕y轴旋转（ry）
        float x4 = cos(ry) * x3 + sin(ry) * z3; 
        float y4 = y3;
        float z4 = -sin(ry) * x3 + cos(ry) * z3;
        //220403can　绕x轴旋转（rx）
        float x5 = x4;
        float y5 = cos(rx) * y4 - sin(rx) * z4;
        float z5 = sin(rx) * y4 + cos(rx) * z4;
        //220403can　绕z轴旋转（rz），再平移
        float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
        float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
        float z6 = z5 + tz;
        //220329 下面就没理解
        //220403can 平移后绕z轴旋转（imuRollStart）
        float x7 = cosImuRollStart * (x6 - imuShiftFromStartX)
                - sinImuRollStart * (y6 - imuShiftFromStartY);
        float y7 = sinImuRollStart * (x6 - imuShiftFromStartX)
                + cosImuRollStart * (y6 - imuShiftFromStartY);
        float z7 = z6 - imuShiftFromStartZ;
        //220329 下面三块，数学原理没理解。依次是绕ｘ，ｙ，ｙ转动？
        //220403can  绕x轴旋转（imuPitchStart）
        float x8 = x7;
        float y8 = cosImuPitchStart * y7 - sinImuPitchStart * z7;
        float z8 = sinImuPitchStart * y7 + cosImuPitchStart * z7;
        //220403can 绕y轴旋转（-imuYawLast）
        float x9 = cosImuYawStart * x8 + sinImuYawStart * z8;
        float y9 = y8;
        float z9 = -sinImuYawStart * x8 + cosImuYawStart * z8;
        //220403can 绕x轴旋转（-imuPitchLast）
        float x10 = cos(imuYawLast) * x9 - sin(imuYawLast) * z9;
        float y10 = y9;
        float z10 = sin(imuYawLast) * x9 + cos(imuYawLast) * z9;
        //220403can 绕z轴旋转（-imuRollLast）
        float x11 = x10;
        float y11 = cos(imuPitchLast) * y10 + sin(imuPitchLast) * z10;
        float z11 = -sin(imuPitchLast) * y10 + cos(imuPitchLast) * z10;
        //220312 从x4...z11注意变量的角度，以及ｘｙｚ之间的赋值规律，但是背后的物理意义，数学原理还没搞懂．
        po->x = cos(imuRollLast) * x11 + sin(imuRollLast) * y11;
        po->y = -sin(imuRollLast) * x11 + cos(imuRollLast) * y11;
        po->z = z11;
        //220403can 只保留线号（表示点的线号，ring激光雷达的第几线的，注意，intensity虚数部分表示的当前点所花的时间）
        po->intensity = int(pi->intensity);     //220312 这里是把点云变换到end时刻的输出点信息
    }
    
    //220330can
    //(rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart,
    // imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz)
    //220330can
    void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly, float blz,    //220312 Imu 旋转插件？计算imu旋转矩阵的
                            float alx, float aly, float alz, float &acx, float &acy, float &acz)    //220312 这里的ac　xyz,带&,目的是获取变换后的返回值的
    {//220312 将输入的角度值，保存到对应的sin,cos值变量，方便后面计算．但是，这里的各个角度变量的物理意义是什么？220401在开始和末尾帧的imu的欧拉角和平移量
    //220331can
    //参考：https://www.cnblogs.com/ReedLW/p/9005621.html
    //                      -imuStart       imuEnd      0
    // transformSum.rot = R             *R          *R
    //                      YXZ             ZXY         k+1
    // bcx, bcy, bcz (rx, ry, rz) 构成了　Ｒ_(k+1)^(0)
    // blx, bly, blz (imuPitchStart, imuYawStart, imuRollStart) 构成了　Ｒ_(YXZ)^(-imuStart)
    // alx, aly, alz (imuPitchLast, imuYawLast, imuRollLast) 构成了　Ｒ_(ZXY)^(imuEnd)
    //220331can
    float sbcx = sin(bcx);
    float cbcx = cos(bcx);
    float sbcy = sin(bcy);
    float cbcy = cos(bcy);
    float sbcz = sin(bcz);
    float cbcz = cos(bcz);

    float sblx = sin(blx);
    float cblx = cos(blx);
    float sbly = sin(bly);
    float cbly = cos(bly);
    float sblz = sin(blz);
    float cblz = cos(blz);

    float salx = sin(alx);
    float calx = cos(alx);
    float saly = sin(aly);
    float caly = cos(aly);
    float salz = sin(alz);
    float calz = cos(alz);
    // 220312 下面的一大堆，目的获取acx,acy,acz,但这里的数学公式的原理是什么？
    float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
                - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
    acx = -asin(srx);

    float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                    - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                    - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                    - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                    + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
    float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                    - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                    - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                    - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                    + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
    acy = atan2(srycrx / cos(acx), crycrx / cos(acx));
    
    float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) 
                    - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx) 
                    - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly) 
                    + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx) 
                    - calx*cblx*cblz*salz) + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz 
                    + sblx*sbly*sblz) + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) 
                    + calx*cblx*salz*sblz);
    float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) 
                    - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx) 
                    + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx) 
                    + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) 
                    + calx*calz*cblx*cblz) - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly 
                    - cbly*sblx*sblz) + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz) 
                    - calx*calz*cblx*sblz);
    acz = atan2(srzcrx / cos(acx), crzcrx / cos(acx));          
    }
    //220403cank　对两帧之间的位姿进行累加，获得相对第一帧的旋转矩阵
    void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz,     //220312 周期积分计算？
                            float &ox, float &oy, float &oz)
    {   //220312 输入变量的物理意义和计算过程的数学公式原理是什么？
        //220401can
        //参考：https://www.cnblogs.com/ReedLW/p/9005621.html
        //０－－－－>(cx, cy, cz)----->(lx, cy, lz) 
        //从０时刻到(cx, cy, cz),然后在(cx, cy, cz)的基础上又旋转(lx, ly, lz)
        //求最后总的位姿结果是什么？
        //　Ｒ*p_cur = R_c*R_l*p_cur ===> R = R_c * R_l
        //（个人理解，R_c这里的ｃ表示camera坐标系，R_l的l表示激光雷达坐标系，p_cur表示当前点
        // 但是疑惑点来了，这里计算的旋转矩阵如何跟这里两个旋转矩阵联系起来？）
        //
        //     |cly*clz+sly*slx*slz  clz*sly*slx-cly*slz  clx*sly|
        // R_l=|    clx*slz                 clx*clz          -slx|
        //     |cly*slx*slz-clz*sly  cly*clz*slx+sly*slz  cly*clx|
        // R_c=...
        // -srx=(ccx*scy,-scx,cly*clx)*(clx*slz,clx*clz,-slx)
        // ...
        //然后根据R来求(ox, oy, oz)
        //220401can（个人还没有理解公式里面的涉及的符号含义）

        float srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx);
        ox = -asin(srx);

        float srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
                    + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy);
        float crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
                    - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx));
        oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

        float srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
                    + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz);
        float crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
                    - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx));
        oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));          
    }

    double rad2deg(double radians)  //弧度转角度
    {
        return radians * 180 / M_PI;
    }

    double deg2rad(double degrees)  //角度转弧度
    {
        return degrees * M_PI / 180 ;
    }

    void findCorrespondingCornerFeatures(int iterCount){    //220312 找相似的角特征？
        
        int cornerPointsSharpNum = cornerPointsSharp->points.size();    // 220312 角点的数目

        for(int i = 0; i < cornerPointsSharpNum; i++){
            
            TransformToStart(&cornerPointsSharp->points[i], &pointSel);     //220312  角点变换到雷达每次扫描开始时刻？处理后的点云变换到每一次扫描的开始时刻，这里pointSel就是保存变换后的点云

            if(iterCount % 5 == 0){ //220312 是５的倍数才计算的理由是什么？220403单次激光点云帧数不超过５，或者任务间隔５帧进行计算处理
                
                kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);    // 220312 kd树检索点
                int closestPointInd = -1,minPointInd2 = -1;     //220312 这里是最近点的id,最小点的id?如何理解？　两个都是表示最近点含义，第一个是计算时使用的初始值，后面是每次遍历的保存值

                 //220403can 该if中变量名中的sq，表示距离的平方值
                 //220403can if 的条件，表示kd树找到一个符合条件的点，则进一步处理
                if(pointSearchSqDis[0] < nearestFeatureSearchSqDist){   //220312 检索到的点比设置的最小距离阈值小，则进行下面操作
                    closestPointInd = pointSearchInd[0];
                    //220403can
                    //在imageProjection.cpp中涉及.intensity的有：
                    //thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
                    //fullInfoCloud->points[index].intensity = range;
                    //所以，这里的intensity保存的是上面二者之一（个人没有明白是点序号还是范围值）
                    //220403can
                    int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity);

                    //220403can 主要功能是找到２个scan之内的最近点，并将找到的最近点及其序号保存
                    //220403can 之前的扫描的保存到minPointSqDis2,之后保存到miniPointSqDis2
                    float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist;
                    for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++){
                        if(int(laserCloudCornerLast->points[j].intensity) > closestPointScan + 2.5){
                            break;  //220312 强度值大于该阈值，则去除．但是，这么考虑的缘由是什么？220404 注意，这里的intensity不是表示强度值，
                                    //而是一个包含该激光点线束号和获取所花费单圈周期相对时间的id参数
                        }

                        pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                    (laserCloudCornerLast->points[j].x - pointSel.x) +
                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                    (laserCloudCornerLast->points[j].y - pointSel.y) +
                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                    (laserCloudCornerLast->points[j].z - pointSel.z) ;  //220312  计算的是两个点的空间上距离，但是pointSel是什么点？表示的是变换到每次扫描开始时刻的点云

                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan){
                            if (pointSqDis < minPointSqDis2){   //220312 找到更靠近的点，更新距离值和ｉｄ
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        } 
                    }//220312 对比上下两个for循环，　分别检索两部分的点云
                    for ( int j = closestPointInd - 1; j >= 0; j--){
                        if(int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5){
                            break;  //220312 强度值不足阈值，则停止循环．物理意义是什么？220404 intensity不是强度值，而是线束号＋相对周期耗时
                        }

                        pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                    (laserCloudCornerLast->points[j].x - pointSel.x) +
                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                    (laserCloudCornerLast->points[j].y - pointSel.y) +
                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                    (laserCloudCornerLast->points[j].z - pointSel.z) ;  //220312  计算的是两个点的空间上距离，但是pointSel是什么点？

                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan){
                            if (pointSqDis < minPointSqDis2){   //220312 找到更靠近的点，更新距离值和ｉｄ
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        } 
                    }
                }

                pointSearchCornerInd1[i] = closestPointInd; //220312 搜索点的ｉｄ?
                pointSearchCornerInd2[i] = minPointInd2;    //220312 距离搜索点的最近的点的id?
            }

            if ( pointSearchCornerInd2[i] >= 0){    //220312 这里更具上面一条注释来推测，逻辑不通．那么，该如何理解？

                tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
                tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

                float x0 = pointSel.x;
                float y0 = pointSel.y;
                float z0 = pointSel.z;
                float x1 = tripod1.x;
                float y1 = tripod1.y;
                float z1 = tripod1.z;
                float x2 = tripod2.x;
                float y2 = tripod2.y;
                float z2 = tripod2.z;
                //220312 这个判断块里面，变量含义和公式意义是什么？
                float m11 = ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1));
                float m22 = ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1));
                float m33 = ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1));

                float a012 = sqrt(m11 * m11  + m22 * m22 + m33 * m33);

                float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                float la =  ((y1 - y2)*m11 + (z1 - z2)*m22) / a012 / l12;

                float lb = -((x1 - x2)*m11 - (z1 - z2)*m33) / a012 / l12;

                float lc = -((x1 - x2)*m22 + (y1 - y2)*m33) / a012 / l12;

                float ld2 = a012 / l12;

                float s = 1;    //220312 这里的s 物理意义是什么，取1的理由是什么？
                if(iterCount >= 5){ //220312 这个判断条件的依据是什么？
                    s = 1 - 1.8 * fabs(ld2);    // 220312 这条公式的依据是什么？
                }

                if (s > 0.1 && ld2 != 0){   //220312 循环条件依据是什么？物理意义是什么？
                    coeff.x = s * la;   //220312 coeff这个变量物理意义是什么？
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    laserCloudOri->push_back(cornerPointsSharp->points[i]); //220312 容器保存的是什么？
                    coeffSel->push_back(coeff);     //220312 这个容器又是放什么？
                }
            }
        }
    }
    
    //220405can 查找面特征和计算回归系数，点离线越远回归系数越低，离线越近的点系数越高
    void findCorrespondingSurfFeatures(int iterCount){  //220313 寻找相关面特征?
        
        int surfPointsFlatNum = surfPointsFlat->points.size();    // 220313 面点的数目

        for(int i = 0; i < surfPointsFlatNum; i++){
            //220405can 转换到起始点
            TransformToStart(&surfPointsFlat->points[i], &pointSel);     //220313  角点变换到开始位置？220403将点云变换到每次扫描开始时刻,处理后的点云保存在pointSel中

            if(iterCount % 5 == 0){ //220313 是５的倍数才计算的理由是什么？
                //220405can 通过i查找j
                    //220313 pointSel表示检索中心点云，1表示检索最近点的个数，pointSearchInd表示检索到的点在点云中的id，pointSearchSqDis表示检索到的点到检索点的距离平方值
                    //220404 函数的作用是在输入点云中找到一个点即可，并返回其在点云中的id值以及距离平方值
                kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);    // 220312 kd树检索点
                int closestPointInd = -1,minPointInd2 = -1, minPointInd3 = -1;     //220313 这里是两个最近点的id,如何理解？
                                    //220404 minPointInd2放的是　
                                        //220313 minPointInd3物理意义是什么？

                 //220403can 该if中变量名中的sq，表示距离的平方值
                 //220403can if 的条件，表示kd树找到一个符合条件的点，则进一步处理
                if(pointSearchSqDis[0] < nearestFeatureSearchSqDist){   //220313 检索到的点比设置的最小距离阈值小，则进行下面操作
                    closestPointInd = pointSearchInd[0];
                    //220403can
                    //在imageProjection.cpp中涉及.intensity的有：
                    //thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
                    //fullInfoCloud->points[index].intensity = range;
                    //所以，这里的intensity保存的是上面二者之一（个人没有明白是点序号还是范围值）
                    //220403can
                    int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity);

                    //220403can
                    //作用是找到２次扫描之间的最近点，并将找到的最近点及其序号保存
                    //小于closestPointScan的保存在minPointSqDis2,其他的保存到minPointSqDis3（220404这里个人没有理解，因为）
                    //220403can
                    float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist, minPointSqDis3 = nearestFeatureSearchSqDist;
                    //220404 对输入点云，从id小到大方向找最近点
                    for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++){
                        if(int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5){
                            break;  //220313 强度值大于该阈值，则去除．但是，这么考虑的缘由是什么？220404 intensity不是强度值，而是线束号＋相对周期耗时
                        }

                        pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                    (laserCloudSurfLast->points[j].x - pointSel.x) +
                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                    (laserCloudSurfLast->points[j].y - pointSel.y) +
                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                    (laserCloudSurfLast->points[j].z - pointSel.z) ;  //220313  计算的是两个点的空间上距离，但是pointSel是什么点？pointSel是经过去畸变处理的点云

                        if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScan){
                            if (pointSqDis < minPointSqDis2){   //220313 找到更靠近的点，更新距离值和id
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }else{
                            if (pointSqDis < minPointSqDis3){ //220313 minPointInd3结合这个变量，这个逻辑判断意义是什么？
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        } 
                    }//22031３ 对比上下两个for循环，　分别检索两部分的点云，上面是id比closestPointInd大的，下面的比closestPointInd小的
                    //220404 从id最大往小方向找最近点
                    //220404can 往前找
                    for ( int j = closestPointInd - 1; j >= 0; j--){
                        if(int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5){//220404 限定查找范围
                            break;  //220313 强度值不足阈值，则停止循环．物理意义是什么？220404 intensity不是强度值，而是该点的雷达线束号+单圈周期相对耗时
                        }
                        //220403 点的距离平方公式
                        pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                    (laserCloudSurfLast->points[j].x - pointSel.x) +
                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                    (laserCloudSurfLast->points[j].y - pointSel.y) +
                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                    (laserCloudSurfLast->points[j].z - pointSel.z) ;    //220313  计算的是两个点的空间上距离，但是pointSel是什么点？
                                                                                        //220403 上面有转换到每次扫描开始时刻函数，这里就是函数返回的点云

                        if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScan){
                            if (pointSqDis < minPointSqDis2){   //220312 找到更靠近的点，更新距离值和ｉｄ
                                minPointSqDis2 = pointSqDis;    //220404 迭代最近阈值
                                minPointInd2 = j;
                            }
                        }else{
                            if (pointSqDis < minPointSqDis3) {
                                minPointSqDis3 = pointSqDis;    //220404 迭代最近阈值
                                minPointInd3 = j;
                            }
                        } 
                    }
                }

                //220405can 查找l和m
                pointSearchSurfInd1[i] = closestPointInd; //220313 搜索点的id?　220404 是的
                pointSearchSurfInd2[i] = minPointInd2;    //220313 距离搜索点的最近的点的id?  220404 在检索的范围内，minPointInd2保存的是距离最小点的id保存在起来
                pointSearchSurfInd3[i] = minPointInd3;    //220313 这里又该如何理解？面点特征为啥比凸点特征多了一个参数？而这个参数的作用又是什么？
                                                            //220404 minPointInd3 保存的是，在检索范围之外，但是符合距离最小的点的id
                                                            //220404 这里的范围是两个区间（０---closestPointInd---surfPointsFlatNum），在０---closestPointInd区间，
                                                            //小于closestPointScan是符合范围的，在区间closestPointInd---surfPointsFlatNum，大于closestPointScan是
                                                            //符合范围的。但是，这么做的意义是什么？
            }

            // 220404can
            // 前后都能找到对应的最近点在给定范围之内
            // 那么就开始计算距离
            // [pa,pb,pc]是tripod1，tripod2，tripod3这3个点构成的一个平面的方向量，
            // ps是模长，它是三角形面积的2倍
            // 220404can
            if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {   //220313 这个判断条件依据是什么？
                //220313 下面的变量意义以及数学公式原理是什么？
                tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
                tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
                tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

                float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z)
                        - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
                float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x)
                        - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
                float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y)
                        - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
                float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

                float ps = sqrt(pa * pa + pb * pb + pc * pc);   //220313 开平方根
                // 220404can 
                // 距离没有取绝对值
                // 两个向量的点乘，分母除以ps中已经除掉了，
                // 加pd原因:pointSel与tripod1构成的线段需要相减
                // 220404can 
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z +pd;    //220313 公式根据是什么？物理意义是什么？

                float s = 1;
                if (iterCount >= 5){    //220313 判断条件依据是什么？220404 激光点云帧数为５？
                    //220404can 影响因子s
                    s = 1 - 1.8 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));  //220313 数学公式的含义是什么？
                }

                // 220405can 计算回归系数
                if(s > 0.1 && pd2 != 0) {   //220313 判断条件依据是什么？
                    //220404can
                    // [x,y,z]是整个平面的单位法量
                    // intensity是平面外一点到该平面的距离
                    //220404can
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;  //220313 根据什么物理意义获取这些公式？

                    //220404can 未经变换的点放入laserCloudOri队列，距离，法向量值放入coeffSel
                    laserCloudOri->push_back(surfPointsFlat->points[i]);    
                    coeffSel->push_back(coeff); //220313 这两个容器是装的是什么？后面用在什么地方？
                }
            }
        }
    }

    //220405can 通过回归系数求解坐标转换Transformation 
    //数学原理查看：https://zhuanlan.zhihu.com/p/384902839
    //220405can 这里使用的是最小二乘法求解
    bool calculateTransformationSurf(int iterCount){    //220313 计算转换面点？

        int pointSelNum = laserCloudOri->points.size(); //220314 点数目
        // 220314 构造一系列的矩阵
        cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));
            // 220314 下面的都是一些数学变量的简化写法，方便下面公式的计算．但公式计算的原理是什么？
        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
        float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
        float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry;

        float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
        float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;
        //220404can  构建雅可比矩阵，求解
        for (int i = 0; i < pointSelNum; i++) { // 220314 循环的目的是把计算结果保存到矩阵中，但是下面几条公式含义是什么？

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (-a1*pointOri.x + a2*pointOri.y + a3*pointOri.z + a4) * coeff.x
                    + (a5*pointOri.x - a6*pointOri.y + crx*pointOri.z + a7) * coeff.y
                    + (a8*pointOri.x - a9*pointOri.y - a10*pointOri.z + a11) * coeff.z;

            float arz = (c1*pointOri.x + c2*pointOri.y + c3) * coeff.x
                    + (c4*pointOri.x - c5*pointOri.y + c6) * coeff.y
                    + (c7*pointOri.x + c8*pointOri.y + c9) * coeff.z;

            float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = arz;
            matA.at<float>(i, 2) = aty;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt); //  220314 计算矩阵matA的转置矩阵，保存在matAt中
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        //220405can 最小二乘法求解
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR); // 220315 求解矩阵方程，结果保存在matX中

        if (iterCount == 0) {
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);  //200315 这里的计算公式如何理解？
            matV.copyTo(matV2);

            isDegenerate = false;   //220315 该控制符号的物理含义是什么？
            float eignThre[3] = {10, 10, 10};   //220315 该变量作用是什么？
            for (int i = 2; i >= 0; i--){   //220315 该循环的作用是什么？条件根据是什么？
                if(matE.at<float>(0, i) < eignThre[i]){
                    for( int j =0; j < 3; j++){
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                }else{
                    break;
                }
            }
            matP = matV.inv() * matV2;  //  220315 matV.inv()矩阵求逆操作
        }

        if(isDegenerate){
            cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;    //220315 把matX转换到到这一步的意义是什么？
        }

        transformCur[0] += matX.at<float>(0, 0);
        transformCur[2] += matX.at<float>(1, 0);
        transformCur[4] += matX.at<float>(2, 0);

        for(int i = 0; i < 6; i++){
            if(isnan(transformCur[i]))      //220315 这么处理的依据是什么？
                transformCur[i] = 0;    
        }

        float deltaR = sqrt(
                        pow(rad2deg(matX.at<float>(0, 0)), 2) +
                        pow(rad2deg(matX.at<float>(1, 0)), 2)); //220315 sqrt 计算的是开平方，pow,计算的是xxx的２次方
        float deltaT = sqrt(
                        pow(matX.at<float>(2, 0) * 100, 2));//220315 上下两个变量分布衡量旋转程度以及平移程度？

        //220405can 结果收敛则返回
        if (deltaR < 0.1 && deltaT < 0.1){  
            return false;
        }
        return true;
    }

    bool calculateTransformationCorner(int iterCount){  //220315 计算角点变换？

        int pointSelNum = laserCloudOri->points.size(); //220315 点数目
        // 220315 构造一系列的矩阵
        cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));
            // 220315 下面的都是一些数学变量的转换，方便下面公式的计算．公式计算的原理是什么？
        //220404can  以下为开始计算A,A=[J的偏导],J的偏导的计算公式是什么?
        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];
        //220315这里的数学公式原理是什么？
        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 -ty*b6 - tx*b5;

        float c5 = crx*srz;

        for (int i = 0; i < pointSelNum; i++) { // 220315 循环的目的是把计算结果保存到矩阵中，但是下面几条公式含义是什么？

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float ary = (b1*pointOri.x + b2*pointOri.y - b3*pointOri.z + b4) * coeff.x
                    + (b5*pointOri.x + b6*pointOri.y - b7*pointOri.z + b8) * coeff.z;

            float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

            float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

            float d2 = coeff.intensity;
            //220404can A=[J的偏导]; B=[权重系数*(点到直线的距离)] 求解公式: AX=B
            //220404can 为了让左边满秩，同乘At-> At*A*X = At*B
            matA.at<float>(i, 0) = ary;
            matA.at<float>(i, 1) = atx;
            matA.at<float>(i, 2) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt); //  220315 计算矩阵matA的转置矩阵，保存在matAt中
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        //220404can 通过QR分解的方法，求解方程AtA*X=AtB，得到X
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR); // 220315 求解矩阵方程，结果保存在matX中

        if (iterCount == 0) {
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

            //220404can 计算At*A的特征值和特征向量
            //220404can 特征值存放在matE，特征向量matV
            cv::eigen(matAtA, matE, matV);  //200315 这里的计算公式如何理解？
            matV.copyTo(matV2);
            //220404can 退化的具体表现是指什么？
            isDegenerate = false;   //220315 该控制符号的物理含义是什么？22040４退化
            float eignThre[3] = {10, 10, 10};   //220315 该变量作用是什么？220404初步理解是一个退化参数
            for (int i = 2; i >= 0; i--){   //220315 该循环的作用是什么？条件根据是什么？
                if(matE.at<float>(0, i) < eignThre[i]){
                    for( int j = 0; j < 3; j++){
                        matV2.at<float>(i, j) = 0;
                    }
                    //220404can 存在比10小的特征值则出现退化
                    isDegenerate = true;
                }else{
                    break;
                }
            }
            matP = matV.inv() * matV2;  //  220315 matV.inv()矩阵求逆操作
        }

        if(isDegenerate){
            cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;    //220315 把matX转换到到这一步的意义是什么？
        }

        transformCur[1] += matX.at<float>(0, 0);
        transformCur[3] += matX.at<float>(1, 0);
        transformCur[5] += matX.at<float>(2, 0);    //220315 transformCur[0,2,4]保存的是calculateTransformationSurf()里面的点，对比的含义是什么？

        for(int i = 0; i < 6; i++){
            if(isnan(transformCur[i]))      //220315 这么处理的依据是什么？
                transformCur[i] = 0;    
        }

        float deltaR = sqrt(
                        pow(rad2deg(matX.at<float>(0, 0)), 2)); //220315 sqrt 计算的是开平方，pow,计算的是xxx的２次方
        float deltaT = sqrt(
                        pow(matX.at<float>(1, 0) * 100, 2) +
                        pow(matX.at<float>(2, 0) * 100, 2));//220315 上下两个变量分布衡量旋转程度以及平移程度？

        if (deltaR < 0.1 && deltaT < 0.1){  
            return false;
        }
        return true;
    }

    bool calculateTransformtion(int iterCount){ //220316 计算整体的变换？基本流程步骤和上面两个计算变换基本一致，但有一个汇总处理的操作．

        int pointSelNum = laserCloudOri->points.size(); //220316 这里和上面的两个都是一样的数量，那么laserCloudOri容器里面的点如何理解？
        // 220316 构造一系列矩阵
        cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        //220316 把cos,sin以及一些大块头使用简化符号代替，方便后面公式编写．transformCur容器里面存放的是什么？
        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];
        //220316 下面公式变量含义以及数学原理是什么？
        float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
        float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
        float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

        float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
        float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

        for (int i = 0; i < pointSelNum; i++) {
            //220316 该循环为了初始化一系列的矩阵，但是数学原理是什么？
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (-a1*pointOri.x + a2*pointOri.y + a3*pointOri.z + a4) * coeff.x
                    + (a5*pointOri.x - a6*pointOri.y + crx*pointOri.z + a7) * coeff.y
                    + (a8*pointOri.x - a9*pointOri.y - a10*pointOri.z + a11) * coeff.z;

            float ary = (b1*pointOri.x + b2*pointOri.y - b3*pointOri.z + b4) * coeff.x
                    + (b5*pointOri.x + b6*pointOri.y - b7*pointOri.z + b8) * coeff.z;

            float arz = (c1*pointOri.x + c2*pointOri.y + c3) * coeff.x
                    + (c4*pointOri.x - c5*pointOri.y + c6) * coeff.y
                    + (c7*pointOri.x + c8*pointOri.y + c9) * coeff.z;

            float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

            float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

            float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = atx;
            matA.at<float>(i, 4) = aty;
            matA.at<float>(i, 5) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);     //220316 计算转置矩阵，保存在matAt
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR); //220316　计算矩阵方程组，结果保存在matX中

        if (iterCount == 0) {   //220316 判断条件依据是什么？
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);  // 220316 这个计算特征向量，保存在matV中
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {10, 10, 10, 10, 10, 10};   //220316 这里数组初始化为10的理由是什么？220404初步理解为一个退化参数设置
            for (int i = 5; i >= 0; i--) {  //220316 取５的缘由是什么？
                if (matE.at<float>(0, i) < eignThre[i]) {   //220316 为啥小于？为啥不直接小于１０，而非得构造一个数组？220404退化参数
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;  //220316 初始化为０的理由什么？
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;  //220316 matV.inv()求逆矩阵，求该变量的含义是什么？或者依据是什么？ 
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformCur[0] += matX.at<float>(0, 0);
        transformCur[1] += matX.at<float>(1, 0);
        transformCur[2] += matX.at<float>(2, 0);
        transformCur[3] += matX.at<float>(3, 0);
        transformCur[4] += matX.at<float>(4, 0);
        transformCur[5] += matX.at<float>(5, 0);    //220316 对早上面的两个函数这部分，其物理意义是什么？

        for(int i=0; i<6; i++){ //220316 这样子初始化作用？
            if(isnan(transformCur[i]))
                transformCur[i]=0;  
        }
        //220404can 计算旋转的模长
        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(rad2deg(matX.at<float>(2, 0)), 2));
        //220404can 计算平移的模长
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));    //220316 上面的公式依据是什么？物理意义是什么？

        if (deltaR < 0.1 && deltaT < 0.1) { //220316 阈值取0.1缘由是什么？
            return false;
        }
        return true;           
    }

    void checkSystemInitialization(){   //220317 检查系统初始化？
        
        //220404can 交换cornerPointsLessSharp和laserCloudCornerLast
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;
        //220317 这两部分交换点云的作用是什么？
        //220404can 交换surfPointsLessFlat和laserCloudSurfLast
        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;
        //220317 kd树操作,设置输入点云
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
        //220317 点云的大小
        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();
        //220317 将pcl点云转为ros类型数据点云
        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2); //220317 发布点云
        //220317 将pcl点云转为ros类型数据点云
        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2); //220317 发布点云
        
        transformSum[0] += imuPitchStart;
        transformSum[2] += imuRollStart;    //220317 两个容器的作用是什么？

        systemInitedLM = true;  //220317 通过这两部分点云发布就表示系统初始化没问题了？
    }

    void updateInitialGuess(){  //220317 更新初始化猜测值？220404 预测位姿
        
        imuPitchLast = imuPitchCur;
        imuYawLast = imuYawCur;
        imuRollLast = imuRollCur;

        imuShiftFromStartX = imuShiftFromStartXCur;
        imuShiftFromStartY = imuShiftFromStartYCur;
        imuShiftFromStartZ = imuShiftFromStartZCur;
        //220317 上下三组变量物理含义是什么？
        imuVeloFromStartX = imuVeloFromStartXCur;
        imuVeloFromStartY = imuVeloFromStartYCur;
        imuVeloFromStartZ = imuVeloFromStartZCur;
        //220404can
        // 关于下面负号的说明：
        // transformCur是在Cur坐标系下的 p_start=R*p_cur+t
        // R和t是在Cur坐标系下的
        // 而imuAngularFromStart是在start坐标系下的，所以需要加负号
        //220404can
        if (imuAngularFromStartX != 0 || imuAngularFromStartY != 0 || imuAngularFromStartZ != 0){
            transformCur[0] = -imuAngularFromStartY;
            transformCur[1] = -imuAngularFromStartZ;
            transformCur[2] = -imuAngularFromStartX;    //220317 这里容器的０，１，２位置和YZX顺序关系如何理解？
        }
        //220404can 速度乘以时间，当前变换中的位移
        if (imuVeloFromStartX != 0 || imuVeloFromStartY != 0 || imuVeloFromStartZ != 0){
            transformCur[3] -= imuVeloFromStartX * scanPeriod;
            transformCur[4] -= imuVeloFromStartY * scanPeriod;
            transformCur[5] -= imuVeloFromStartZ * scanPeriod;  //220317 这几个容器装得是什么？后面用来干什么？
        }
    }

    //220405 特征匹配的数学原理，可以查看这大佬的https://zhuanlan.zhihu.com/p/384902839
    void updateTransformation(){    //220317 更新变换？ 220405 特征匹配更合理

        if (laserCloudCornerLastNum < 10 || laserCloudSurfLastNum < 100)
            return;     //220317 角点数量小于１０或者面点小于１００，直接退出
        
        //220405can 面特征匹配
        for (int iterCount1 = 0; iterCount1 < 25; iterCount1++){    //220317 这个循环的目地获取什么？
            laserCloudOri->clear();
            coeffSel->clear();

            //220404can
            // 找到对应的特征平面
            // 然后计算协方差矩阵，保存在coeffSel队列中
            // laserCloudOri中保存的是对应于coeffSel的未转换到开始时刻的原始点云数据
            //220404can
            //220405can 查找匹配的特征
            findCorrespondingSurfFeatures(iterCount1);    

            if (laserCloudOri->points.size() < 10)
                continue;
            //220404can 通过面特征的匹配，计算变换矩阵
            //220405can 计算Transformation
            if(calculateTransformationSurf(iterCount1) == false)
                break;
        }

        //220405can 线特征匹配
        for(int iterCount2 = 0; iterCount2 < 25; iterCount2++) {    //220317 这个循环的目地获取什么？

            laserCloudOri->clear();
            coeffSel->clear();

            //220404can 
            // 找到对应的特征边/角点
            // 寻找边特征的方法和寻找平面特征的很类似，过程可以参照寻找平面特征的注释
            //220404can
            //220405can 查找匹配的特征
            findCorrespondingCornerFeatures(iterCount2);

            if(laserCloudOri->points.size() < 10)
                continue;
            //220404can 通过角/边特征的匹配，计算变换矩阵
            //220405can 计算Transformation
            if(calculateTransformationCorner(iterCount2) == false)
                break;
        }
    }
    //220405can 特征提取实际上是提取面特征和线特征，然后利用最小二乘法做特征匹配，得
    //到当前最优的坐标转换关系


    //220404can 旋转角的累计变化量
    void integrateTransformation(){     //220317 合并变换？
        float rx, ry, rz, tx, ty, tz;
        
        //220404can
        // AccumulateRotation作用
        // 将计算的两帧之间的位姿“累加”起来，获得相对于第一帧的旋转矩阵
        // transformSum + (-transformCur) =(rx,ry,rz)
        //220404can
        AccumulateRotation(transformSum[0], transformSum[1], transformSum[2],
                            -transformCur[0], -transformCur[1], -transformCur[2], rx, ry, rz);
        
        //220404can 进行平移分量的更新
        float x1 = cos(rz) * (transformCur[3] - imuShiftFromStartX) 
                - sin(rz) * (transformCur[4] - imuShiftFromStartY);
        float y1 = sin(rz) * (transformCur[3] - imuShiftFromStartX) 
                + cos(rz) * (transformCur[4] - imuShiftFromStartY);
        float z1 = transformCur[5] - imuShiftFromStartZ;

        float x2 = x1;
        float y2 = cos(rx) * y1 - sin(rx) * z1;
        float z2 = sin(rx) * y1 + cos(rx) * z1;

        tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);
        ty = transformSum[4] - y2;
        tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);
        // 220317 上面的变量含义是什么？其中的物理意义以及数学原理是什么？
        //220404can 与accumulateRotatio联合起来更新transformSum的rotation部分的工作
        //220404can 可视为transformToEnd的下部分的逆过程
        PluginIMURotation(rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart, 
                        imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz);

        transformSum[0] = rx;
        transformSum[1] = ry;
        transformSum[2] = rz;
        transformSum[3] = tx;
        transformSum[4] = ty;
        transformSum[5] = tz;       // 220317 这几个的物理意义是什么？存起来后面有什么用途？
    }

    void publishOdometry(){     //220317 发布里程计
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformSum[2], -transformSum[0], -transformSum[1]);
        //220404can rx,ry,rz转化为四元数发布
        laserOdometry.header.stamp = cloudHeader.stamp;
        laserOdometry.pose.pose.orientation.x = -geoQuat.y;
        laserOdometry.pose.pose.orientation.y = -geoQuat.z;
        laserOdometry.pose.pose.orientation.z = geoQuat.x;
        laserOdometry.pose.pose.orientation.w = geoQuat.w;  //220317 将一个四元数信息赋予里程计,那么，这个四元数信息物理意义是什么？
        laserOdometry.pose.pose.position.x = transformSum[3];
        laserOdometry.pose.pose.position.y = transformSum[4];
        laserOdometry.pose.pose.position.z = transformSum[5];
        pubLaserOdometry.publish(laserOdometry);    //220317 发布里程计信息
        //220317 下面这几步目的是干什么？220404用于广播tf
        //220404can laserOdometryTrans 是用于tf广播
        laserOdometryTrans.stamp_ = cloudHeader.stamp;
        laserOdometryTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        laserOdometryTrans.setOrigin(tf::Vector3(transformSum[3], transformSum[4], transformSum[5]));
        tfBroadcaster.sendTransform(laserOdometryTrans);
    }

    void adjustOutlierCloud(){  // 220317 调整outlier外部？点云，进行这一步的作用是什么？
        PointType point;
        int cloudSize = outlierCloud->points.size();
        for (int i = 0; i < cloudSize; ++i) //220317 这一步是在没理解，把这个点导出来，然后又填回原来位置，意义是什么？
        {
            point.x = outlierCloud->points[i].y;
            point.y = outlierCloud->points[i].z;
            point.z = outlierCloud->points[i].x;
            point.intensity = outlierCloud->points[i].intensity;
            outlierCloud->points[i] = point;    
        }
    }

    void publishCloudsLast(){   //220317 发布点云

        updateImuRollPitchYawStartSinCos();

        int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
        for (int i = 0; i < cornerPointsLessSharpNum; i++){ //220317 循环意义是什么？
        //220404can TransformToEnd的作用是将k+1时刻的less特征点转移至k+1时刻的sweep的结束位置处的雷达坐标系下
            TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
        }


        int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
        for (int i = 0; i < surfPointsLessFlatNum; i++){    //220317 循环意义是什么？
            TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
        }

        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;
        // 220317 上下这里的交换点内容，作用是什么？
        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;    

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();

        if ( laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {     //220317 条件性设置kdtree，物理意义是什么？
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
        }

        frameCount++;

        if ( frameCount >= skipFrameNum + 1){   // 220318 skipframeNum物理意义是什么？

            frameCount = 0; //220318 重置为０的理由是什么？
            //220404can 调整坐标系，x=y,y=z,z=x
            adjustOutlierCloud();
            sensor_msgs::PointCloud2 outlierCloudLast2;
            pcl::toROSMsg(*outlierCloud, outlierCloudLast2);
            outlierCloudLast2.header.stamp = cloudHeader.stamp;
            outlierCloudLast2.header.frame_id = "/camera";
            pubOutlierCloudLast.publish(outlierCloudLast2);

            sensor_msgs::PointCloud2 laserCloudCornerLast2;
            pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
            laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
            laserCloudCornerLast2.header.frame_id = "/camera";
            pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

            sensor_msgs::PointCloud2 laserCloudSurfLast2;
            pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
            laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
            laserCloudSurfLast2.header.frame_id = "/camera";
            pubLaserCloudSurfLast.publish(laserCloudSurfLast2);                       
        }
    }

    //220405can 主流程
    //220405can 特征提取，提取线特征和面特征
    //特征提取的过程：先对点云进行畸变校正（运动补偿），接着计算点的平滑程度，然后按照平滑度排序，
    //如果是不平滑的点，则选为线特征（柱子或者墙壁的棱角），如果是平滑的点，则选为面特征（地面，
    //墙面等平面）。同时为了避免选择的特征过于集中在同一个地方，会把360°方向(俯视角度看)切分为6个区域，每个
    //区域平均选择2个线特征和4个面特征。

    //220405can 特征匹配，利用最小二乘法，获取当前最优的位姿
    void runFeatureAssociation()    //220318 运行特征提取？
    {
        //220404can 如果有新数据进来而且消息延迟小于0.05秒，则执行，否则不执行任何操作 
        if (newSegmentedCloud && newSegmentedCloudInfo && newOutlierCloud &&
            std::abs(timeNewSegmentedCloudInfo - timeNewSegmentedCloud) < 0.05 &&
            std::abs(timeNewOutlierCloud - timeNewSegmentedCloud) < 0.05){  //220318 一个是新的，另外一个条件是时间间隔够短

            newSegmentedCloud = false;
            newSegmentedCloudInfo = false;
            newOutlierCloud = false;
        }else{
            return;
        }

        // 1. feature extraction 特征提取

        //220404can 主要进行的处理是将点云数据进行坐标变换，进行插补等工作
        //220405can 点云运动补偿
        adjustDistortion();
        //220404can 不完全按照公式进行光滑性计算，并保存结果
        //220405can 计算平滑度
        caluclateSmoothness();
        //220404can 标记阻塞点??? 阻塞点是什么点???
        //220404can 参考了csdn若愚maimai大佬的博客，这里的阻塞点指过近的点
        //220404can 指在点云中可能出现的互相遮挡的情况
        //220405can 标记遮挡点
        markOccludedPoints();
        //220404can 特征抽取，然后分别保存到cornerPointsSharp等等队列中去
        //220404can 保存到不同的队列是不同类型的点云，进行了标记的工作，
        //220404can 这一步中减少了点云数量，使计算量减少
        //220405can 提取特征
        extractFeatures();
        //220404can 发布cornerPointsSharp等4种类型的点云数据
        //220405can 发布点云
        publishCloud(); // cloud for visualizetion  发布点云目的是可视化


        //2. feature association 特征融合(匹配)

        if(!systemInitedLM){
            //220405can　检查系统初始化
            checkSystemInitialization();
            return;
        }
        //220404can 预测位姿
        //220405can　更新初始猜测位置
        updateInitialGuess();
        //220404can 更新变换
        //220405can　特征匹配，输出Transformation
        updateTransformation();
        //220404can 积分总变换
        //220405can　变换坐标Transformation
        integrateTransformation();
        //220404 发布激光里程计信息
        //220405can　发布雷达里程计
        publishOdometry();
        //220404　发布最新帧激光点云
        //220405can　发布点云用于建图
        publishCloudsLast();    //cloud to mapOtimization 发布的点云目的给mapOptimization模块建图
    }
};




int main(int argc,char** argv)
{
    ros::init(argc, argv, "logo_loam");

    ROS_INFO("\033[1;32m--->\033[0m Feature Association Started.");

    FeatureAssociation FA;

    ros::Rate rate(200);
    while(ros::ok())
    {
        ros::spinOnce();

        FA.runFeatureAssociation();

        rate.sleep();
    }

    ros::spin();
    return 0;
}