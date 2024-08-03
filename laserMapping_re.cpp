#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include "IMU_Processing.hpp"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
int add_point_size = 0, kdtree_delete_counter = 0;
bool pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;            //条件变量（condition_variable）是 C++ 中用于多线程同步的一种机制。它通常与互斥锁（mutex）一起使用，用于实现线程间的等待和通知机制。
ros::Publisher pubLaserCloudMap;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int scan_count = 0, publish_count = 0;
int feats_down_size = 0, NUM_MAX_ITERATIONS = 0, pcd_index = 0;

bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> init_pos(3, 0.0);
vector<double> init_rot{0, 0, 0, 1};
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());  //畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); //畸变纠正后降采样的单帧点云，W系
PointCloudXYZI::Ptr cloud(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;

esekfom::esekf kf;                                                    //存储我们待优化的状态量

state_ikfom state_point;
Eigen::Vector3d pos_lid; //估计的W系下的位置

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);        //同时，通过调用 notify_all() 来通知其他线程或者条件变量，以触发相应的后续处理。
    sig_buffer.notify_all();              //一个 ROS 节点中用于处理信号，当接收到特定信号时，设置退出标记并输出日志。


}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)         //每一个scan的点云
{
    mtx_buffer.lock();                                                       //先锁上，防止其他线程调用
    scan_count++;                                                            //计数加一
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);                                               //把msg经过process处理，转化为pcl类型的点云 ptr即处理好的点云
    lidar_buffer.push_back(ptr);                                            //ptr即pl_surf[128]
    time_buffer.push_back(msg->header.stamp.toSec());                       //时间戳buff
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();                                                    //多线程解锁
    sig_buffer.notify_all();                                                //唤醒其他线程
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;                                                                                  //计数加一
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)                                     //判断外部的时间同步是否开启，若开启加上偏移量，雷达与imu的时间差，到与雷达时间对齐
    {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());       //.toSec() 获取秒数  .fromSec()将输入的秒数转化为ros时间
    }

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);  //确定时间戳

    double timestamp = msg->header.stamp.toSec();                                                    //重新定义时间戳

    mtx_buffer.lock();                                                                //互斥锁 来获取锁，表明当前线程需要对某个共享资源进行保护，确保在加锁期间其他线程无法访问该资源，从而保证线程安全性

    if (timestamp < last_timestamp_imu)                                               //如果这次imu的时间戳小于上一次的时间戳，有问题
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;                                                   //更新时间戳

    imu_buffer.push_back(msg);                                                        //将imu信息push进imu_buffer里
    mtx_buffer.unlock();                                                              //将锁解开，和下一行结合使用，用于多线程等待和唤醒
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
//把当前要处理的LIDAR和IMU数据打包到meas
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)                                                         //判断雷达点是否被push过了
    {
        meas.lidar = lidar_buffer.front();                                     //将lidar_buffer里的第一个点给meas.lidar
        meas.lidar_beg_time = time_buffer.front();                             //将time_buffer里的第一个点给meas.lidar_beg_time
        if (meas.lidar->points.size() <= 5) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;        //雷达平均扫描时间
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)   //小于0.5倍的平均扫描时间
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num++;                                                              //scan 加一
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;                                       //储存时间

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)                                     //判断最新的imu时间戳是否小于雷达最后的时间戳
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))                //收集两个雷达帧之间的imu数据 并将imu_buffer里的点传入meas.imu队列中
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());                 
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

BoxPointType LocalMap_Points;      // ikd-tree地图立方体的2个角点
bool Localmap_Initialized = false; // 局部地图是否初始化
void lasermap_fov_segment()
{
    cub_needrm.clear(); // 清空需要移除的区域
    kdtree_delete_counter = 0;

    V3D pos_LiD = pos_lid; // W系下位置
    //初始化局部地图范围，以pos_LiD为中心,长宽高均为cube_len
    if (!Localmap_Initialized)       //如果是第一次初始化
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    //各个方向上pos_LiD与局部地图边界的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（1.5*300m）太小，标记需要移除need_move(FAST-LIO2论文Fig.3) det_range 激光雷达探测的距离
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return; //如果不需要，直接返回，不更改局部地图

    BoxPointType New_LocalMap_Points, tmp_boxpoints;  //更新地图开始了 新的局部地图点、临时立方体角点
    New_LocalMap_Points = LocalMap_Points;
    //需要移动的距离
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)               //向左移动
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)          //向右移动
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);

    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm); //删除指定范围内的点
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I.matrix() * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

//根据最新估计位姿  增量添加点云到map
void init_ikdtree()
{
    //加载读取点云数据到cloud中   //ROOT_DIR 可能是一个宏定义或全局变量，用于表示根目录的路径
    string all_points_dir(string(string(ROOT_DIR) + "PCD/") + "GlobalMap_ikdtree.pcd");    //"GlobalMap_ikdtree.pcd" 是点云数据文件的名称,"PCD/" 是一个相对目录，用于指明存储点云数据的子目录
    if (pcl::io::loadPCDFile<PointType>(all_points_dir, *cloud) == -1)  //使用 PCL（Point Cloud Library）的函数 loadPCDFile 从指定路径加载点云数据文件到 cloud 中
    {
        PCL_ERROR("Read file fail!\n");   //如果加载失败（返回值为 -1），则输出错误信息。
    }

    ikdtree.set_downsample_param(filter_size_map_min);          //设置降采样参数
    ikdtree.Build(cloud->points);                               //根据加载的点云构建ikd树
    std::cout << "---- ikdtree size: " << ikdtree.size() << std::endl;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher &pubLaserCloudFull_)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyToWorld(&laserCloudFullRes->points[i],
                             &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull_.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyToWorld(&feats_undistort->points[i],
                             &laserCloudWorld->points[i]);
        }

        static int scan_wait_num = 0;
        scan_wait_num++;

        if (scan_wait_num % 4 == 0)
            *pcl_wait_save += *laserCloudWorld;
    }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);

    auto q_ = Eigen::Quaterniond(state_point.rot.matrix());
    out.pose.orientation.x = q_.coeffs()[0];
    out.pose.orientation.y = q_.coeffs()[1];
    out.pose.orientation.z = q_.coeffs()[2];
    out.pose.orientation.w = q_.coeffs()[3];
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";                          //将里程计信息储存在odomAftMapped中
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);

    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;     //transformBroadcaster()类就是一个publisher,在实际的使用中，我们需要在某个Node中构建tf::TransformBroadcaster类
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));   //调用sendTransform(),将transform发布到/tf的一段transform上。   StampedTransform()：一个带有时间戳信息的坐标系变换类，包含了四元数、平移向量和时间戳等信息，可以用于表示一个坐标系相对于另一个坐标系的姿态和位置关系
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");                                   //通过ros获取参数
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);            // 是否发布当前正在扫描的点云的topic
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);          // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true); // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，需要该变量和上一个变量同时为true才发布
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);                   // 卡尔曼滤波的最大迭代次数
    nh.param<string>("map_file_path", map_file_path, "");                    // 地图保存路径
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");         // 雷达点云topic名称
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");           // IMU的topic名称
    nh.param<bool>("common/time_sync_en", time_sync_en, false);              // 是否需要时间同步，只有当外部未进行时间同步时设为true
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5); // VoxelGrid降采样时的体素大小    Voxel：体素 Grid：网格
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);     //寻找最近邻时的体素立方体大小
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);    // 地图的局部区域的长度（FastLio2论文中有解释）
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f); // 激光雷达的最大探测范围
    nh.param<double>("mapping/fov_degree", fov_deg, 180);            //fov:视场   degree 视场角
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);               // IMU陀螺仪的协方差
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);               // IMU加速度计的协方差
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);        // IMU陀螺仪偏置的协方差
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);        // IMU加速度计偏置的协方差
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);        // 最小距离阈值，即过滤掉0～blind范围内的点云
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA); // 激光雷达的类型
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);       // 激光雷达扫描的线数（livox avia为6线）
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);        //unit表示单位
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);           // 采样间隔，即每隔point_filter_num个点取1个点 体素滤波
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false); // 是否提取特征点（FAST_LIO2默认不进行特征点提取）
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false); // 是否将点云地图保存到PCD文件
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>()); // 雷达相对于IMU的外参R
    nh.param<vector<double>>("mapping/init_pos", init_pos, vector<double>()); // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    nh.param<vector<double>>("mapping/init_rot", init_rot, vector<double>()); // 雷达相对于IMU的外参R

    cout << "Lidar_type: " << p_pre->lidar_type << endl;
    // 初始化path的header（包括时间戳和帧id），path用于保存odemetry的路径
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);               //面点云下采样，设置下采样栅格大小
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);                   //地图点下采样，

    shared_ptr<ImuProcess> p_imu1(new ImuProcess());                                                                //实例化共享指针 p_imu1指针指向imuprocess
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);                                                                     //两个外参的输入
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    //开始给imu一个初值
    p_imu1->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov),
                      V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, b_acc_cov));                  //将参数全部输入到ImuProcess类里

    signal(SIGINT, SigHandle);                   //当程序检测到signal信号（例如ctrl+c）时，执行SigHandle函数 信号处理  //flg_exit=true
    ros::Rate rate(5000);

    init_ikdtree(); //读取点云文件 初始化ikdtree
    //初始化状变量
    state_point = kf.get_x();
    state_point.pos = Eigen::Vector3d(init_pos[0], init_pos[1], init_pos[2]);
    Eigen::Quaterniond q(init_rot[3], init_rot[0], init_rot[1], init_rot[2]);
    Sophus::SO3 SO3_q(q);
    state_point.rot = SO3_q;
    kf.change_x(state_point);

    while (ros::ok())
    {
        if (flg_exit)
            break;
        ros::spinOnce();                                           //ROS节点处理消息队列中的消息，执行相应的回调函数，并立即返回，使得节点能够及时响应外部消息和事件。

        if (sync_packages(Measures))                               //把一次的IMU和LIDAR数据打包到Measures
        {
            double t00 = omp_get_wtime();

            if (flg_first_scan)                                            //判断是不是第一帧雷达点云
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu1->first_lidar_time = first_lidar_time;                //将第一帧雷达时间赋给imu处理函数中的雷达时间                               
                //将加载的点云数据文件转换为 ROS (Robot Operating System) 中定义的 sensor_msgs::PointCloud2 数据类型，以便在 ROS 系统中进行后续处理或传输
                string all_points_dir(string(string(ROOT_DIR) + "PCD/") + "GlobalMap.pcd");//表示点云数据文件的完整路径,定义了一个名为 all_points_dir 的字符串变量，并初始化为拼接字符串的结果
                if (pcl::io::loadPCDFile<PointType>(all_points_dir, *cloud) == -1)  //这部分代码使用 PCL（Point Cloud Library）的函数 loadPCDFile 从指定路径加载点云数据文件到 cloud 中。
                {
                    PCL_ERROR("Read file fail!\n");                                 //如果加载失败（返回值为 -1），则输出错误信息
                }

                sensor_msgs::PointCloud2 laserCloudMap; //声明了一个 ROS 中的 sensor_msgs::PointCloud2 类型的变量 laserCloudMap，用于存储转换后的点云数据。
                pcl::toROSMsg(*cloud, laserCloudMap);  //调用 PCL 提供的函数 toROSMsg 将 PCL 格式的点云数据 cloud 转换为 ROS 中的 sensor_msgs::PointCloud2 类型，存储在 laserCloudMap 中
                laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
                laserCloudMap.header.frame_id = "camera_init";
                pubLaserCloudMap.publish(laserCloudMap);    //将转换后的点云数据 laserCloudMap 发布到 ROS 中的一个话题（topic）上，以便其他节点可以订阅这个话题并接收这些点云数据

                flg_first_scan = false;
                continue;
            }

            p_imu1->Process(Measures, kf, feats_undistort);

            //如果feats_undistort为空 ROS_WARN
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;                   //雷达的原点在世界坐标系下的位置

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            lasermap_fov_segment(); //更新localmap边界，然后降采样当前帧点云

            //点云下采样
            downSizeFilterSurf.setInputCloud(feats_undistort);                //输入是补偿后的一帧点云
            downSizeFilterSurf.filter(*feats_down_body);                      //输出下采样之后的点云
            feats_down_size = feats_down_body->points.size();                 //拿到点云的大小数目

            // std::cout << "feats_down_size :" << feats_down_size << std::endl;
            if (feats_down_size < 5)                                          //点太少了，抛出一个警告
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
             //初始化ikdtree(ikdtree为空时)
            if (ikdtree.Root_Node == nullptr)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
                feats_down_world->resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // lidar坐标系转到世界坐标系，第一个参数为输入，第二个参数为输出
                }
                ikdtree.Build(feats_down_world->points); //根据世界坐标系下的点构建ikdtree
                continue;
            }

            if (0) // If you need to see map point, change to "if(1)" 是否要看全局地图
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
                std::cout << "ikdtree size: " << featsFromMap->points.size() << std::endl;
            }

            /*** iterated state estimation ***/
            Nearest_Points.resize(feats_down_size); //存储近邻点的vector  resize 调整
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            feats_down_world->resize(feats_down_size);
            map_incremental();                         //地图的增量更新

            /******* Publish points *******/
            if (path_en)
                publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)
                publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body);

            double t11 = omp_get_wtime();
            std::cout << "feats_down_size: " << feats_down_size << "  Whole mapping time(ms):  " << (t11 - t00) * 1000 << std::endl
                      << std::endl;
        }

        rate.sleep();
    }

    return 0;
}
