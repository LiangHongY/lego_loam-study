
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

#include "utility.h"

class ImageProjection{
    private:

        ros::NodeHandle nh;

        ros::Subscriber subLaserCloud;

        ros::Publisher pubFullCloud;
        ros::Publisher pubFullInfoCloud;

        ros::Publisher pubGroundCloud;
        ros::Publisher pubSegmentedCloud;
        ros::Publisher pubSegmentedCloudPure;
        ros::Publisher pubSegmentedCloudInfo;
        ros::Publisher pubOutlierCloud;
        
        pcl::PointCloud<PointType>::Ptr laserCloudIn;
        pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;

        pcl::PointCloud<PointType>::Ptr fullCloud;  //projected velodyne raw cloud,but saved in the form of 1-D matrix
        pcl::PointCloud<PointType>::Ptr fullInfoCloud;  // same as fullCloud,but with intensity - range

        pcl::PointCloud<PointType>::Ptr groundCloud;
        pcl::PointCloud<PointType>::Ptr segmentedCloud;
        pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
        pcl::PointCloud<PointType>::Ptr outlierCloud;

        PointType nanPoint; //fill in fullCloud at each iteration

        cv::Mat rangeMat;   //range matrix for range image　距离图片的距离矩阵
        cv::Mat labelMat;   //label matrix for segmentaiton marking　分割时的标签矩阵
        cv::Mat groundMat;  //ground matrix for ground cloud marking   地面点云的矩阵
        int labelCount;

        float startOrientation;
        float endOrientation;

        cloud_msgs::cloud_info segMsg;  //info of segmented cloud
        std_msgs::Header cloudHeader;

        std::vector<std::pair<int8_t,int8_t>> neighborIterator; //neighbor iterator for segmentaiton process

        uint16_t *allPushedIndX;    //array for tracking points of a segmented object
        uint16_t *allPushedIndY;

        uint16_t *queueIndX;    //array for breadth-first search process of segmentation ,for speed
        uint16_t *queueIndY;

    public:
        ImageProjection():
            nh("~"){
                //220220ros订阅器，订阅激光雷达数据topic  pointCloudTopic = "/velodyne_points"
                //220220lego-loam开源时，采用的是velodyne厂商激光雷达，所以topic写成这样子，如果雷达属于其他家的，只要数据结构一致，内容稍作修改，通过topic映射就可以跑这个框架
                //220405 输入点云回调，一切的开始（输入点云－点云分割－特征提取（面、边特征）－特征匹配－输出位姿－点云注册－回环检测＋生成全局地图－地图优化）
                subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);

                //220405 发布分割好的点云（含地面点云）
                pubFullCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_projected", 1);
                pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_info", 1);

                pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
                pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 1);
                pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2>("segmented_cloud_pure", 1);
                pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info>("segmented_cloud_info", 1);
                pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud", 1);

                //220405 对nan类型的点进行统一赋值
                nanPoint.x = std::numeric_limits<float>::quiet_NaN();
                nanPoint.y = std::numeric_limits<float>::quiet_NaN();
                nanPoint.z = std::numeric_limits<float>::quiet_NaN();
                nanPoint.intensity = -1;    

                allocateMemory();   //分配内存
                resetParameters();  //重置参数（初始化参数）
            }
        
        //初始化各类参数以及分配内存
        void allocateMemory(){
                laserCloudIn.reset(new pcl::PointCloud<PointType>());   //这些语句如何理解？
                laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());

                fullCloud.reset(new pcl::PointCloud<PointType>());
                fullInfoCloud.reset(new pcl::PointCloud<PointType>());

                groundCloud.reset(new pcl::PointCloud<PointType>());
                segmentedCloud.reset(new pcl::PointCloud<PointType>());
                segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
                outlierCloud.reset(new pcl::PointCloud<PointType>());

                fullCloud->points.resize(N_SCAN*Horizon_SCAN);
                fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

                segMsg.startRingIndex.assign(N_SCAN, 0);
                segMsg.endRingIndex.assign(N_SCAN, 0);

                segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
                segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
                segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

                //含义？
                //220220can labelComponents函数使用该矩阵
                //220220can　该矩阵用于求某个点的上下左右４个邻接点
                std::pair<int8_t, int8_t> neighbor;
                neighbor.first = -1; neighbor.second = 0; neighborIterator.push_back(neighbor);
                neighbor.first = 0; neighbor.second = 1; neighborIterator.push_back(neighbor);
                neighbor.first = 0; neighbor.second = -1; neighborIterator.push_back(neighbor);
                neighbor.first = 1; neighbor.second = 0; neighborIterator.push_back(neighbor);

                allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];  //开辟一个数组空间？
                allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

                queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
                queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
            }

        //220220can 初始化／重置各类参数内容
        void resetParameters(){

                //是把原始参数配置清除吗？220405 是的，清除之前的参数
                laserCloudIn->clear();
                groundCloud->clear();
                segmentedCloud->clear();
                segmentedCloudPure->clear();
                outlierCloud->clear();

                //参数含义是啥？220405 这里是初始化矩阵，里面放着一系列的点
                rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX)); //220220点云集合？220405 不是，是range image 距离图片的矩阵
                groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));   //220220地面点云集合?220405　地面点云矩阵
                labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));   //220220标记点云集合？ 220405 点云标签时使用的标签矩阵
                labelCount = 1; //标记数？　220405点云簇数（聚类数目），这里初始化为１

                std::fill(fullCloud->points.begin(), fullCloud->points.end(),nanPoint);     //202202202220这个内连函数如何理解？处理点云的作用是啥？
                std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
            }

        ~ImageProjection(){}

        //220219复制点云？ 220405,是的，但是传递点云来理解更合适
        void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
                //220220can 将ROS中的sensor_msgs::PointCloud2ConstPtr类型转换到pcl点云指针
                cloudHeader = laserCloudMsg->header;
                //cloudHeader.stamp = ros::Time::now(); //ouster lidar users may need to nucomment this line
                pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);     //ros自带函数，将ros的点云转为pcl点云

                //remove nan points剔除点云中nan点
                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
                //have "ring" channel in the cloud
                if(useCloudRing == true){//220405can 如果点云有"ring"通过，则保存为laserCloudInRing
                    pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);
                    if(laserCloudInRing->is_dense == false){    //220214　这个判断条件含义是啥？ 220405 false表示输入的点云是不稠密的
                        ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                        ros::shutdown();
                    }
                }
            }

        //220214这个函数很关键，介绍了很多的函数含义
        void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
            //1. convert ros message to pcl point cloud 将ros消息转换成pcl点云
            copyPointCloud(laserCloudMsg);
            //2. start and end angle of a scan  扫描的开始和结束角度
            findStartEndAngle();
            //3. range image projection     //220220　激光图片范围？   220404点云投影到距离图像
            projectPointCloud();
            //4. mark ground points     //220220　地面特征点云？220404 标记地面点云
            groundRemoval();
            //5. point cloud segmentation   //220220 激光点云分割
            //220404can 思想：
            // 在去除地面之后，对接下来的点进行分割。这里通过BFS（深度优先遍历）递归进行查找，
            //从[0,0]点开始，遍历它前、后、左、右的4个点，分别进行对比，如果相对角度大于60°，
            //则认为是同一个点云集群。最后分割出来的点云数量大于30个则认为分割有效（实际上大于
            //5个可能也行）。
            cloudSegmentation();
            //6.publish all clouds          //220220　发布所有点云（分割后的）
            publishCloud();
            //7. reset parameters for next iteration    //220220　为下一次迭代重置参数
            resetParameters();
        }
        
        //220405can 查找起始和终止角度（激光雷达旋转的起始终止角度）
        void findStartEndAngle(){
            //220220can
            //激光雷达坐标系：右－＞Ｘ，前－＞Ｙ，上－＞Ｚ
            //激光雷达内部旋转方向：Ｚ轴俯视下来，顺时针方向

            //atan2(y,x)函数的返回值范围[－pi,pi],表示复数x+yi的辐角
            //segMsg.startOrientation范围（－pi,pi］
            //segMsg.endOrientation范围（pi,3pi］
            //因为内部雷达旋转方向缘由，所以atan2()函数需要添加一个负号
            //220220can
            //start and end orientation of this cloud
            segMsg.startOrientation = -atan2(laserCloudIn->points[0].y,laserCloudIn->points[0].x);
            segMsg.endOrientation = -atan2(laserCloudIn->points[laserCloudIn->points.size() -1].y,
                                            laserCloudIn->points[laserCloudIn->points.size() - 1].x )+ 2*M_PI;
                    //220220can
                    // 开始和结束的角度差一般是多少？
                    //一个velodyne雷达数据包转过的角度多大？
                    //雷达一般包含的是一圈的数据，所以角度差一般是2*pi,也就是说一个数据包涵盖的是360度
                    //220220can
                //segMsg.endOrientation - segMsg.startOrientation范围为(0,4pi]（个人没有理解为啥是4pi220221）
                //如果角度差大于３pi或者小于pi，说明角度差有问题，需要进行调整
                //220221can    
            if(segMsg.endOrientation - segMsg.startOrientation > 3*M_PI){
                segMsg.endOrientation -= 2*M_PI;
            }else if(segMsg.endOrientation - segMsg.startOrientation < M_PI){
                segMsg.endOrientation += 2*M_PI;
            }
            //220221can segMsg.orientationDiff 的范围为（pi,3pi），一圈大小为３pi，应该在２pi左右（这句话个人没有理解）
            segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
        }

        void projectPointCloud(){
            // range image projection   220404　点云映射成图像
            //220405can 将点云理解为球，将球投影到圆柱体侧面，侧面展开就成点云图像
            float verticalAngle, horizonAngle, range;
            size_t rowIdn, columnIdn, index, cloudSize;
            PointType thisPoint;

            cloudSize = laserCloudIn->points.size();    //220405 点云的容量
            //220405 遍历点云
            for(size_t i = 0; i< cloudSize; ++i){
                thisPoint.x = laserCloudIn->points[i].x;
                thisPoint.y = laserCloudIn->points[i].y;
                thisPoint.z = laserCloudIn->points[i].z;
                //find the row and column index in the iamge for this point　找到图片中点的行列索引号
                if(useCloudRing == true){//220404can 如果有ring index则直接采用
                    rowIdn = laserCloudInRing->points[i].ring;  //220214 ring含义？ 220405 点云类型里面一个属性
                }
                else{   //220404can 如果没有ring index,则通过计算角度得到index
                    //220221can 计算竖直方向上的角度（雷达的第几线（这句话没有理解））
                    //220404 多线激光雷达的竖直方向上，不同线的角度不一样的，所以可以根据角度进行推导第几线
                    verticalAngle = atan2(thisPoint.z,sqrt(thisPoint.x*thisPoint.x + thisPoint.y*thisPoint.y))*180/M_PI;
                    //220221can rowIdn计算出该点激光雷达是竖直方向上的第几线
                    //220221can 从下往上计数，  -15度记为初始线，第０线，一共N_SCAN线（如N_SCAN = 16）
                    rowIdn = (verticalAngle + ang_bottom) / ang_res_y;  //220405 纵坐标
                }
                if(rowIdn < 0 || rowIdn >= N_SCAN){
                    continue;
                }
                //220221 atan2 反正切  sqrt 开平方 round 四舍五入取整
                //220221can
                //atan2(y,x)函数的返回值范围(-pi,pi),表示与复数x+yi的幅角
                //下方角度atan2()交换了x和y的位置，计算的是与y轴正方向的夹角大小(关于y = x做对称变换)
                //这里是在雷达坐标系，所有是与正前方的夹角大小
                //220221can
                horizonAngle = atan2(thisPoint.x, thisPoint.y)*180/M_PI;

                    //220221can
                    //round函数进行四舍五入取整
                    //这边不是减去１８０度？不是
                    //雷达水平方向上某个角度和水平第几线的关联关系如下：
                    //horizonAngle:(-pi,pi],columnIdn:[H/4,5H/4]--->[0,H](H:Horizon_SCAN)
                    //下面是把坐标系绕ｚ轴旋转，对columnIdn进行线性变换
                    //x +==> Horizon_SCAN/2,    x -===>Horizon_SCAN
                    //y +==> Horizon_SCAN*3/4,  y -==>Horizon_SCAN*5/4,Horizon_SCAN/4

            //
            //          3/4*H
            //          | y+
            //          |
            // (x-)H---------->H/2 (x+)
            //          |
            //          | y-
            //    5/4*H   H/4
            //
                    //220221can (个人表示没有理解坐标系啥意思 220405 x表示水平，y表示垂直方向？)
                //220404can 计算横坐标
                columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN/2;
                if(columnIdn >= Horizon_SCAN){
                    columnIdn -= Horizon_SCAN;
                }  
                //220221can
                // 经过上面columnIdn -= Horizon_SCAN的变换后的columnIdn分布：
                //          3/4*H
                //          | y+
                //     H    |
                // (x-)---------->H/2 (x+)
                //     0    |
                //          | y-
                //         H/4
                //
                //220221can
                if(columnIdn < 0|| columnIdn >= Horizon_SCAN){
                    continue;
                }

                //220222 range表示激光点到激光雷达的距离，若小于最小设定值１米，则中断返回
                range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
                if(range < sensorMinimumRange){
                    continue;
                }
                //220404 将距离值保存到矩阵中
                rangeMat.at<float>(rowIdn, columnIdn) = range;
                //220222can columnIdn:[0,H](H:Horizon_SCAN)==>[0,1800](这里的０到1800如何理解？220404 1800表示单圈扫描得到的激光点数)
                //220404can 这里的强度值，为了保存横、纵坐标值
                thisPoint.intensity = (float)rowIdn +(float)columnIdn / 10000.0;
                //220222columnIdn和rowIdn有啥关系含义？rowIdn表示的是第几线，那么columnIdn是啥含义?
                //220404can 把点云保存到数组，fullCloud的强度为横纵坐标，fullInfoCloud中的为距离
                index = columnIdn + rowIdn*Horizon_SCAN;
                fullCloud->points[index] = thisPoint;
                fullInfoCloud->points[index] = thisPoint;   //220222这里跟fullCloud一样内容，具体含义是啥？单纯是显示点信息？
                fullInfoCloud->points[index].intensity = range; //the corresponding range of a point is saved as "intensity"把点的距离保存为点的强度
            }
        }
        
        //去除地面，如果一个点跟相邻的点成角小于10度，则判断为地面点
        //220404 这里的点成角，可以参考https://blog.csdn.net/nh54zyt/article/details/116028175#t8
        void groundRemoval(){
            size_t lowerInd, upperInd;
            float diffX, diffY, diffZ,  angle;
            //groundMat
            // -1, no valid info to check if ground of not表示不是检查是否是地面的有效信息
            // 0, initial value, after validation, means not ground初始化数值，并不表示地面
            // 1, ground表示地面
            for(size_t j=0; j < Horizon_SCAN; ++j){
                //groundScanInd是在　utility.h 文件中声明的线数，groundScanInd = 10
                for(size_t i=0; i < groundScanInd; ++i ){
                    //220405 获取两个点，判断是否地面点
                    lowerInd = j + (i)*Horizon_SCAN;
                    upperInd = j + (i+1)*Horizon_SCAN;


                    //220222can 初始化用nanPoint.intensity = -1　填充
                    //都是-1,证明是空点nanPoint
                    if(fullCloud->points[lowerInd].intensity == -1 ||
                        fullCloud->points[upperInd].intensity == -1){
                            //no info to check,invalid points没有信息校核，无效点
                            groundMat.at<int8_t>(i,j) = -1;
                            continue;
                        }
                            //220222can 由上下两线之间点的XYZ位置得到两线之间的俯仰角
                            //如何俯仰角在10度以内，则判断（i,j）为地面点，groundMat[i][j] = 1
                            //220222can 否则，则不是地面点，进行后续操作
                    diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                    diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                    diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;
                    //220405can 计算两个点的成角，结果把弧度转为度
                    angle = atan2(diffZ,sqrt(diffX*diffX + diffY*diffY))*180/M_PI;
                    if(abs(angle - sensorMountAangle) <=10){ //220405 当成角小于10度，表示属于地面点   
                        groundMat.at<int8_t>(i,j) = 1;//220215左侧的表达式含义是啥？可以理解为一个数组，保存的是地面点信息
                        groundMat.at<int8_t>(i+1,j) =1;//220222为啥把(i+1,j)也视为地面点? 220405 这时候把另外一个点也是地面点 
                    }
                }
            }
            // extract ground cloud (groundMat == 1)    220215提取地面点云
            // mark entry that doesn't need to label (ground and invalid point) for segmentation标识出地面点和无效点用于分割
            // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
            //注意地面剔除是从０～Ｎ_SCAN-1,需要构建一个距离标签矩阵给16线（若使用的是128线，则是给128线距离标签矩阵）
                //220222can 找到所有点中的地面点或者距离为FLT_MAX(rangeMat的初始值)的点，并将他们标记为-1
                //220222can rangeMat[i][j] == FLT_MAT,代表的含义是什么？无效点
            //220405 如果为地面或者rangeMat为空，则标记为-1,后面去除这些点
            for(size_t i = 0;i < N_SCAN; ++i){
                for(size_t j = 0; j < Horizon_SCAN; ++j){
                    if(groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                        labelMat.at<int>(i,j) = -1;
                    }
                }
            }

            //220223can 若有节点订阅groundCloud,那么就把地面点发布出去(getNumSubscribers()该成员函数是ros里面发布者自带的？？？？)
            //循环就是把地面点放到groundCloud队列里面
            if(pubGroundCloud.getNumSubscribers() != 0){
                for(size_t i = 0; i <= groundScanInd; ++i){
                    for(size_t j = 0; j< Horizon_SCAN; ++j){
                        if(groundMat.at<int8_t>(i,j) == 1){ //220223这里＝１表示就是地面点
                            groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        }
                    }
                }
            }
        }

        //220217点云分割
        //220404can
        //去除地面点后，对剩余的点进行分割。这里采用BFS（深度优先遍历）递归进行查找，从[0,0]点开始，
        //遍历它前后左右四个点，分别进行对比，如果相对角度大于60度，则认为是同一个点云集群，最后分割
        //出来的点云数量大于30个，则认为点云分割有效（实际上大于５个也行）
        //（这里的相对角度大于60，如何理解？属于激光雷达的经验知识吗？）
        //220404can
        void cloudSegmentation(){
            //segmentation process　分割过程
            for(size_t i = 0;i < N_SCAN; ++i){
                for(size_t j = 0; j < Horizon_SCAN; ++j){
                            //220223can
                            //如果labelMat[i][j] = 0，表示没有对该点进行过分类
                            //需要对该点进行聚类（聚类是啥意思？220404 同一类型的点，打上标签分类，处理时对不同种类可以采用不同的处理方式）
                            //聚类是把相似的对象通过静态分类的方法分成不同的组别或者更多的子集(来自维基百科)
                            //220223can
                    if(labelMat.at<int>(i,j) == 0){ //220405 若点云还没有标记，则进行标记操作
                        labelComponents(i, j);  //220404　对点云进行了标记
                    }
                }
            }
            //220404 下面过程就是标记好的点云进行分割
            int sizeOfSegCloud = 0;
            //extract segmented cloud for lidar odometry    220223提取分割点云作为雷达的odom???
            for(size_t i = 0; i < N_SCAN; ++i){
                        //220223
                        //segMsg.startRingIndex[i]
                        //segMsg.endRingIndex[i]
                        //表示第i线的点云起始序列和终止序列
                        //以开始线后的第６线开始，以结束线前的第６线为结束（结合该层循环来看，似乎解析了-1 +/- 5 的含义，但是个人没有理解）
                segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;//220215如何理解

                for(size_t j = 0; j< Horizon_SCAN; ++j){
                            //220223can 找到可用的特征点或者地面点（labelＭat[i][j] = 0 的点）
                    if(labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                        //outliers that will not be used for optimization (always continue)极端值不会用来进行优化的
                            //220223can
                            //labelMat数值999999表示这个点因为聚类数量不足30而被舍弃的点（个人好奇这个根据是在哪里的？220404 人为约定）
                            //需要舍弃的点直接continue跳过本次循环
                            //当列数为５的倍数，并且行数较大，可以认为非地面点，将它保存进异常点云（界外点云）中
                            //然后跳过本次循环
                        if(labelMat.at<int>(i,j) == 999999){//220215去除极端值　220405 标签为999999则跳过
                            if(i > groundScanInd && j % 5 ==0){
                                outlierCloud->push_back(fullCloud->points[j + i* Horizon_SCAN]);
                                continue;//220215保存的是异端值？那为啥保存呢？
                            }else{
                                continue;
                            }
                        }
                        //majority of ground points are skipped 地面点直接跳过
                        //220224can 如果是地面点，对于列数（index）不为5的倍数的，直接跳过
                        if(groundMat.at<int>(i,j) == 1){    //220223判断为地面点
                            if(j%5 != 0 && j>5 && j<Horizon_SCAN -5){//220215这个判断的含义是啥？
                                                                    //对于列数非５倍数并且非始末５范围内都进行跳过处理
                                continue;
                            }
                        }

                        //220224can 上面多个if语句已经去掉了不符合条件的点，这部分直接进行信息的拷贝和保存操作
                        //mark ground points so they will not be considered as edge features later
                        //标志地面点，以免视为边沿特征
                        segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);

                        //mark the points' column index for marking occlusion later
                        //220215?给点簇的每个点添加id
                        segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                        //save range info
                        //220215? 保存激光点距离信息
                        segMsg.segmentedCloudRange[sizeOfSegCloud] = rangeMat.at<float>(i,j);
                        //save seg cloud保存分割点云
                        segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        //size of seg cloud
                        ++sizeOfSegCloud;
                    }
                }
                
                //220224can 以结束线前的第５线为结束
                segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;  //220215这里的-1是啥意思？直接－６不行吗？结合前面解析，对于末尾５个点的数据都直接剔除意思
                                                                //220405 可以这么理解，-1就是一个数组的最大id号意思
            }                                       
                        //220224can 如果有节点订阅SegmentedCloudPure,
                        //220224can 那么把点云数据保存到SegmentedCloudPure
            //extract segmented cloud for visualization 获取分割点云进行可视化220217
            //220405can 这里的分割点云，不含地面
            if(pubSegmentedCloudPure.getNumSubscribers() != 0){//220224 ros的类Publisher含有成员函数getNumSubscribers()，其返回的是发布者当前所拥有的订阅者数目
                for(size_t i = 0; i < N_SCAN; ++i){
                    for(size_t j = 0; j < Horizon_SCAN; ++j){
                        if(labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                            segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);   //220217填充点云
                            segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);    //220217给点云添加的是啥？反射率？颜色？应该是反射率
                        }
                    }
                }
            }
        }
        
        //220218打标签组件？ 220405 给点云打标签 
        void labelComponents(int row,int col){
            //use std::queue std::vector std::deque will slow the program down greatly
            //使用标准容器queue,vector会让程序运行的速度大大减低
            float d1, d2,alpha, angle;
            int fromIndX, fromIndY, thisIndX, thisIndY;
            bool lineCountFlag[N_SCAN] = {false};   //220217一个布尔数组，并且并且成员都赋值为false

            queueIndX[0] = row;     //220217以下定义变量的含义是啥？
            queueIndY[0] = col;
            int queueSize = 1;
            int queueStartInd = 0;
            int queueEndInd = 1;

            allPushedIndX[0] = row;
            allPushedIndY[0] = col;
            int allPushedIndSize = 1;   //220217为啥给１？变量名称含义是啥？

            //220224can
            //标准的BFS(广度优先搜索算法，一种暴力搜索算法)
            //BFS的作用是以（row,col）为中心向外面扩散
            //判断(row,col)是否是这个平面中的一点

            //220405 第一个点为[row,col],长度为１。依据看上面初始化
            while (queueSize > 0)
            {
                //Pop point220217出点
                fromIndX = queueIndX[queueStartInd];
                fromIndY = queueIndY[queueStartInd];
                --queueSize;
                ++queueStartInd;
                //Mark popped point标记已经出来的点
                        //220226 在初始化函数resetParameters()中labelCount初始化为１，后面会递增
                labelMat.at<int>(fromIndX,fromIndY) = labelCount;

                //loop through all the neighboring grids of popped grid 220217如何理解？
                        //220226can neighor = [[-1,0];[0,1];[0,-1];[1,0]](这四个点是如何得到的？220404　算法根据点的位置，自动检索该点的前后左右四个点)
                        //220226can 遍历点［fromIndX,fromIndY］边上的四个邻点
                        //220404can 遍历前后左右四个点同时，也计算角度差
                for(auto iter = neighborIterator.begin(); iter != neighborIterator.end();++iter){//220224 遍历相邻点
                    //new index 220217新目录
                    thisIndX = fromIndX + (*iter).first;
                    thisIndY = fromIndY + (*iter).second;

                    //index should be within the boundary  220217目录带界？是这么理解吗？
                    if(thisIndX < 0 || thisIndY >= N_SCAN){     //220226　id超出范围的点都直接跳过
                        continue;
                    }
                    //at range image margin (left or right side)
                    //220226can 环状图片，左右连通（点云和图片啥关系？指的是点云图片？）
                    if(thisIndY < 0){
                        thisIndY = Horizon_SCAN - 1;
                    }
                    if(thisIndY >= Horizon_SCAN){//220226 单圈点id超过最大的1800，则重置归零
                        thisIndY = 0;   
                    }
                    //prevent infinite loop (防止无限循环)(caused by put already examined point back)
                    if(labelMat.at<int>(thisIndX, thisIndY) != 0){  //220218判断条件含义如何理解？220226　这里表示标记数值为０时，直接跳过
                                                                    //由于［thisIndX,thisdY］为标记过的点，并且在labelMat里面，－１表示无效点，
                                                                    // ０表示未标记，其他均表示已标记
                                                                    // 220226can　如果labelMat已经标记过正整数，则已经聚类完成，不需要再次对该点聚类
                        continue;
                    }
                    //220218 how to knowledge d1 and d2 ?
                    // 220226 计算相邻点(thisIndX, thisIndY)到点(fromIndX,fromIndY)的最值
                    d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                                    rangeMat.at<float>(thisIndX, thisIndY));
                    d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                                    rangeMat.at<float>(thisIndX, thisIndY));
                        //220226can alpha代表角度分辨率
                        //　Ｘ方向上角度分辨率是segmentAlphaX(rad)
                        // Y方向上角度分辨率是segmentAlphaY(rad)
                        //220405 x为横向，y表示纵向
                    if((*iter).first == 0){
                        alpha = segmentAlphaX;
                    }else{
                        alpha = segmentAlphaY;
                    }

                    // 220226can 通过下面公式计算两点之间是否有平面特征
                    // atan2(y,x)的值越大，d1,d2之间的差距越小，表示平面越平坦
                    angle = atan2(d2*sin(alpha),(d1 - d2*cos(alpha)));

                    if(angle > segmentTheta){
                                                // 220226 segmentTheta在utility.h里面有定义声明，角度为60度，但计算时转为弧度制
                                            // 220226can 如果算出的角度大于60度，那么假设其为一个平面（220405 这里跟点云分割原理理解有点出入，若点云分割原理，其表示非同一物体的点云）
                        queueIndX[queueEndInd] = thisIndX;  //220218thisIndX如何理解？表示点的ｘ坐标
                        queueIndY[queueEndInd] = thisIndY;
                        ++queueSize;
                        ++queueEndInd;

                        labelMat.at<int>(thisIndX, thisIndY) = labelCount;  //220218如何理解？点标签id
                        lineCountFlag[thisIndX] = true;     //220218如何理解？

                        allPushedIndX[allPushedIndSize] = thisIndX;
                        allPushedIndY[allPushedIndSize] = thisIndY;
                        ++allPushedIndSize;
                    }
                }
            }
            
            //check if this segment is valid检查分割是有效的（检查聚类是否有效）
            bool feasibleSegment = false;

            // 
            if(allPushedIndSize >= 30){
                    // 220226can　如果聚类点集的点数目超过30，则表示是一个可用聚类，labelCount需要递增
                feasibleSegment = true;
            }//220404can 点大于5个，并且横跨3个纵坐标，这时候也认为聚类成功
            else if(allPushedIndSize >= segmentValidPointNum){//220218判断条件如何理解？
                                // segmentValidPointNum在utility.h里面有定义声明　=5　
                                // 220226can　如果聚类点集的点数目小于30大于等于5，统计竖直方向的聚类点数
                int lineCount = 0;
                for(size_t i = 0; i < N_SCAN; ++i){
                    if(lineCountFlag[i] == true){   
                        ++lineCount;
                    }
                }
                if(lineCount >= segmentValidLineNum){
                    // segmentValidLineNum在utility.h有声明定义　＝３,220226can 当数目超过３个时候，也标记为有效聚类
                    feasibleSegment = true;
                }
            }
            
            //segment is valid ,mark these points
            // 220226 聚类有效，标记这些聚类（簇）
            if(feasibleSegment == true){
                ++labelCount;   //220404can labelCount代表分割为了多少个簇
            }else{
                //segment is invalid, mark these points
                // 220226can 标记为999999的聚类点，是需要舍弃的，因为它们的数目小于30
                //220405 聚类无效，标记为999999
                for(size_t i = 0; i < allPushedIndSize; ++i){
                    labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
                }
            }
        }

        //220218发布点云
        void publishCloud(){
            //1.publish seg cloud info
            // 发布cloud_msgs::cloud_info消息（聚类后点云信息） 
            //220405can 发布点云，包括采样后的地面和分割后的点云消息
            segMsg.header = cloudHeader;
            pubSegmentedCloudInfo.publish(segMsg);
            //2.publish clouds发布OutlierCloud界外点云
            //220405can 发布离群后的点云
            sensor_msgs::PointCloud2 laserCloudTemp;

            pcl::toROSMsg(*outlierCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubOutlierCloud.publish(laserCloudTemp);
            //segmented cloud with ground发布SegmentedCloud分块（聚类）点云
            //220405can 发布点云，包括采样后的地面和分割后的点云
            pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloud.publish(laserCloudTemp);
            //projected full cloud
            //220226 发布点云（聚类＋地面）信息，注意和下面发布的info区别开
            //220405can 发布的是投影到距离图片的点云
            if(pubFullCloud.getNumSubscribers() != 0){
                pcl::toROSMsg(*fullCloud, laserCloudTemp);
                laserCloudTemp.header.stamp = cloudHeader.stamp;
                laserCloudTemp.header.frame_id = "base_link";
                pubFullCloud.publish(laserCloudTemp);
            }
            //origrial dense ground cloud
            // 220226 如果有订阅地面点云的话题存在，则发布地面点云
            if(pubGroundCloud.getNumSubscribers() != 0){
                pcl::toROSMsg(*groundCloud, laserCloudTemp);
                laserCloudTemp.header.stamp = cloudHeader.stamp;
                laserCloudTemp.header.frame_id = "base_link";
                pubGroundCloud.publish(laserCloudTemp);
            }
            //segmented cloud without ground
            // 220226 如果有人订阅聚类点云（不包含地面点云），则发布
            if(pubSegmentedCloudPure.getNumSubscribers() != 0){
                pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
                laserCloudTemp.header.stamp = cloudHeader.stamp;
                laserCloudTemp.header.frame_id = "base_link";
                pubSegmentedCloudPure.publish(laserCloudTemp);
            }
            //project full cloud info
            //220226 发布所有点云（聚类＋地面）附带的信息
            if(pubFullInfoCloud.getNumSubscribers() != 0){
                pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
                laserCloudTemp.header.stamp = cloudHeader.stamp;
                laserCloudTemp.header.frame_id = "base_link";
                pubFullInfoCloud.publish(laserCloudTemp);
            }
        }
};

int main(int argc,char** argv){
    ros::init(argc, argv, "logo_loam");
    ImageProjection IP;
    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");//220219含义？

    ros::spin();
    return 0;
}




