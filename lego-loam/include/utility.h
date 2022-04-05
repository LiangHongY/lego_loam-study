#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include"cloud_msgs/cloud_info.h"

// #include <opencv/cv.h>
#include <opencv2/imgproc.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#define PI 3.14159265

using namespace std;

typedef pcl::PointXYZI PointType;

extern const string pointCloudTopic = "/velodyne_points";   //输入激光点云话题
extern const string imuTopic = "/imu/data";     //mapOptimization 里面的一个Imu话题名称

//save pcd
extern const string fileDirectory = "/tmp/";    //保存pcd的路径

//using velodyne cloud 'ring' channel for image projection 
//(other lidar may have different name for this channel, change "PointXYZIR" below)
//false表示不使用ring(velodye支持，其他看雷达厂商)
extern const bool useCloudRing = false;  //if true,ang_res_y and ang_bottom are not used

//VLP-128
extern const int N_SCAN = 128;          //220223激光雷达线数
extern const int Horizon_SCAN = 1800;   //220223单圈点云的点数
extern const float ang_res_x = 0.2;     //水平分辨率
extern const float ang_res_y = 0.3;     //垂直分辨率
extern const float ang_bottom = 25.0;   //lidar　底部线束角度偏移
extern const int groundScanInd = 10;    //lidar　判断为地面的线数

extern const bool loopClosureEnableFlag = false;   //false表示不使用回环检测 
extern const double mappingProcessInterval = 0.3;   //建图时间间隔/周期

extern const float scanPeriod = 0.1;    //激光扫描周期时间
extern const int systemDelay = 0;       //系统延迟
extern const int imuQueLength = 200;    //imu缓存大小

extern const float sensorMinimumRange = 1.0;    //过滤激光雷达为中心的１米范围内的点云
extern const float sensorMountAangle = 0.0;     //传感器安装角度？
//点云分割方向角度（这里的60度分割原理，个人没有理解）
extern const float segmentTheta = 60.0/180.0*M_PI;    //decrease this value may improve accuracy　在imageProjection里面判断是两块点云的关系，如果两点成角小于60度,则认为属于同一物体的点云，否则不属于
extern const int segmentValidPointNum = 5;              //点云分割有效点数。聚类有效点判断阈值参数（imageProjection）
extern const int segmentValidLineNum = 3;               //点云分割有效线数。表示聚类的竖直方向聚类的点数阈值参数（imageProjection）
//本代码中是将激光点云投影到相机坐标系的，ｘ水平，ｙ上下，ｚ前后
extern const float segmentAlphaX = ang_res_x / 180.0 * M_PI;    // 水平方向最小分辨率。Ｘ方向上角度分辨率(弧度制)（imageProjection）
extern const float segmentAlphaY = ang_res_y / 180.0 * M_PI;     // 垂直方向最小分辨率。Y方向上角度分辨率（弧度制）（imageProjection）              
                       

extern const int edgeFeatureNum = 2;    //线特征数量
extern const int surfFeatureNum = 4;    //面特征数量
extern const int sectionsTotal = 6;     //界面合计数，代码中并没有用到
extern const float edgeThreshold = 0.1; //线曲率阈值
extern const float surfThreshold = 0.1; //面曲率阈值
extern const float nearestFeatureSearchSqDist = 25; //特征查找最小半径

// mapping params
//地图搜索半径大小
extern const float surroundingKeyframeSearchRadius = 50.0;// key frame that is within n meters from current pose will be considerd for scan-to-map optimization (when loop closure disabled)
//地图点云搜索数量大小
extern const int surroundingKeyframeSearchNum = 50; //submap size (when loop closure enabled)
//history key frames (history submap for loop closure)
//回环检测首尾距离（回环检测检索半径）
extern const float historyKeyframeSearchRadius = 7.0;    // key frame that is within n meters from current pose will be considerd for loop closure
//回环点云查找范围？（回环检查点云检索帧数）
extern const int historyKeyframeSearchNum = 25; //// 2n+1 number of hostory key frames will be fused into a submap for loop closure 在mapOptmization模块中使用
//icp匹配分数，平均距离小于0.3
extern const float historyKeyframeFitnessScore = 0.3;   // the smaller the better alignment
//可视化点云范围（全局可视化地图检索半径）
extern const float globalMapVisualizationSearchRadius = 500.0;  /// key frames with in n meters will be visualized

struct  smoothness_t
{
    float value;
    size_t ind;
};

struct by_value
{
    bool operator()(smoothness_t const &left, smoothness_t const &right){
        return left.value < right.value;
    }
};
// A point cloud type that has "ring" channel
//定义带ring的点云类型
struct PointXYZIR
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR,
                                (float, x, x) (float, y, y)
                                (float, z, z) (float, intensity, intensity)
                                (uint16_t, ring, ring)
)

// A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
//点云带有６个姿态信息
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;//疑惑点，大写字母为啥后面不需要添加分号的.答：220220 这个是eigen库里面的一个宏定义

//这是什么编写结构？220405 理解为库里面宏定义
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time)
)

typedef PointXYZIRPYT PointTypePose;

#endif


























