#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{
struct Rgb
{
  float r;
  float g;
  float b;
};

// HSV 转 RGB 的简单实现
Rgb hsvToRgb(float h, float s, float v)
{
  h = std::fmod(h, 360.0f);
  if (h < 0.0f) {
    h += 360.0f;
  }

  const float c = v * s;
  const float x = c * (1.0f - std::fabs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
  const float m = v - c;

  float rp = 0.0f;
  float gp = 0.0f;
  float bp = 0.0f;

  if (h < 60.0f) {
    rp = c;
    gp = x;
  } else if (h < 120.0f) {
    rp = x;
    gp = c;
  } else if (h < 180.0f) {
    gp = c;
    bp = x;
  } else if (h < 240.0f) {
    gp = x;
    bp = c;
  } else if (h < 300.0f) {
    rp = x;
    bp = c;
  } else {
    rp = c;
    bp = x;
  }

  return Rgb{rp + m, gp + m, bp + m};
}
}  // namespace

class ClusterRvizNode final : public rclcpp::Node
{
public:
  ClusterRvizNode() : rclcpp::Node("cluster_rviz")
  {
    pcd_path_ = this->declare_parameter<std::string>("pcd_path", "table_scene_lms400.pcd");
    frame_id_ = this->declare_parameter<std::string>("frame_id", "map");

    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 1.0);

    voxel_leaf_size_ = this->declare_parameter<double>("voxel_leaf_size", 0.02); // 体素滤波的叶子大小
    plane_distance_threshold_ = this->declare_parameter<double>("plane_distance_threshold", 0.06); // 平面分割的距离阈值
    plane_max_iterations_ = this->declare_parameter<int>("plane_max_iterations", 100);

    cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.1); // 聚类距离阈值（需要大于voxel_leaf_size_）
    min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 100);
    max_cluster_size_ = this->declare_parameter<int>("max_cluster_size", 25000);

    enable_crop_box_ = this->declare_parameter<bool>("enable_crop_box", false);
    crop_min_ = this->declare_parameter<std::vector<double>>("crop_min", std::vector<double>{-3.0, -3.0, -3.0});
    crop_max_ = this->declare_parameter<std::vector<double>>("crop_max", std::vector<double>{3.0, 3.0, 3.0});

    publish_filtered_cloud_ = this->declare_parameter<bool>("publish_filtered_cloud", true);

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("input_cloud", qos);
    if (publish_filtered_cloud_) {
      filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_cloud", qos);
    }
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("cluster_boxes", qos);

    // 记录输出：record/cluster/<启动时间>.csv
    initRecording();

    const auto period = std::chrono::duration<double>(1.0 / std::max(0.1, publish_rate_hz_));
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { this->computeAndPublish(); });
  }

private:
  struct TrackUpdateResult
  {
    std::vector<Eigen::Vector2d> filtered_xy;
    std::vector<int> track_ids;
  };

  struct Kalman2D
  {
    Eigen::Vector4d x = Eigen::Vector4d::Zero();
    Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
    bool initialized = false;

    void init(double px, double py)
    {
      x << px, py, 0.0, 0.0;
      P.setIdentity();
      P(0, 0) = 1.0;
      P(1, 1) = 1.0;
      P(2, 2) = 10.0;
      P(3, 3) = 10.0;
      initialized = true;
    }

    void predict(double dt, double process_noise)
    {
      if (!initialized) {
        return;
      }

      Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
      F(0, 2) = dt;
      F(1, 3) = dt;
      x = F * x;

      const double q = process_noise;
      const double dt2 = dt * dt;
      const double dt3 = dt2 * dt;
      const double dt4 = dt2 * dt2;
      Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
      Q(0, 0) = dt4 / 4.0 * q;
      Q(0, 2) = dt3 / 2.0 * q;
      Q(1, 1) = dt4 / 4.0 * q;
      Q(1, 3) = dt3 / 2.0 * q;
      Q(2, 0) = dt3 / 2.0 * q;
      Q(2, 2) = dt2 * q;
      Q(3, 1) = dt3 / 2.0 * q;
      Q(3, 3) = dt2 * q;

      P = F * P * F.transpose() + Q;
    }

    void update(double meas_x, double meas_y, double meas_noise)
    {
      if (!initialized) {
        init(meas_x, meas_y);
        return;
      }

      Eigen::Matrix<double, 2, 4> H;
      H.setZero();
      H(0, 0) = 1.0;
      H(1, 1) = 1.0;

      Eigen::Vector2d z;
      z << meas_x, meas_y;

      Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * meas_noise;

      const Eigen::Vector2d y = z - H * x;
      const Eigen::Matrix2d S = H * P * H.transpose() + R;
      const Eigen::Matrix<double, 4, 2> K = P * H.transpose() * S.inverse();

      x = x + K * y;
      const Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
      P = (I - K * H) * P;
    }

    double px() const { return x(0); }
    double py() const { return x(1); }
  };

  struct Track
  {
    int id = 0;
    Kalman2D kf;
    rclcpp::Time last_predict_stamp; // 上次预测的时间戳（用于计算预测时的 dt）
    rclcpp::Time last_update_stamp;  // 上次更新的时间戳（用于判断 track 是否在本帧被更新）
    int missed = 0;
  };

  void initRecording()
  {
    namespace fs = std::filesystem;

    // 文件名按“程序启动时间”生成
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    start_time_str_ = oss.str();

    fs::path dir = fs::path("record") / "cluster";
    std::error_code ec;
    fs::create_directories(dir, ec);

    record_path_ = (dir / (start_time_str_ + ".csv")).string();
    record_stream_.open(record_path_, std::ios::out | std::ios::app);

    // 按“帧”记录：一帧一行，robots 字段里并排写多个 robot
    record_stream_ << "stamp_sec,stamp_nanosec,robots\n";
    record_stream_.flush();

    RCLCPP_INFO(this->get_logger(), "Kalman记录文件: %s", record_path_.c_str());
  }

  int createTrack(double x, double y, const rclcpp::Time & stamp)
  {
    Track tr;
    tr.id = next_track_id_++;
    tr.kf.init(x, y);
    tr.last_predict_stamp = stamp;
    tr.last_update_stamp = stamp;
    tr.missed = 0;
    tracks_.push_back(tr);
    return tr.id;
  }

  TrackUpdateResult updateTracks(const std::vector<Eigen::Vector2d> & measurements, const rclcpp::Time & stamp)
  {
    constexpr double kGateDistance = 1.0;         // 匹配门限：测量点与 track 预测点距离超过这个值就不匹配
    constexpr int kMaxMissed = 200;                // 超过这个帧数未匹配就删除 track
    constexpr double kProcessNoise = 2.0;         // 调大：允许更快运动
    constexpr double kMeasurementNoise = 0.05;    // 调大：更不信任观测

    // 返回：每个 measurement 对应的 Kalman 输出 (x,y)
    TrackUpdateResult result;
    result.filtered_xy.assign(measurements.size(), Eigen::Vector2d::Zero());
    result.track_ids.assign(measurements.size(), -1);

    // 1) predict
    for (auto & tr : tracks_) {
      double dt = 1.0 / std::max(0.1, publish_rate_hz_);
      if (tr.last_predict_stamp.nanoseconds() > 0) {
        dt = (stamp - tr.last_predict_stamp).seconds();
        if (!(dt > 0.0)) {
          dt = 1.0 / std::max(0.1, publish_rate_hz_);
        }
      }
      tr.kf.predict(dt, kProcessNoise);
      tr.last_predict_stamp = stamp;
    }

    // 2) associate (greedy nearest neighbor)
    std::vector<int> meas_to_track(measurements.size(), -1);
    std::vector<bool> track_used(tracks_.size(), false);

    for (std::size_t mi = 0; mi < measurements.size(); ++mi) {
      const auto & z = measurements[mi];
      double best_d = std::numeric_limits<double>::infinity();
      int best_ti = -1;
      for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
        if (track_used[ti]) {
          continue;
        }
        const double dx = tracks_[ti].kf.px() - z.x();
        const double dy = tracks_[ti].kf.py() - z.y();
        const double d = std::sqrt(dx * dx + dy * dy);
        if (d < best_d) {
          best_d = d;
          best_ti = static_cast<int>(ti);
        }
      }
      if (best_ti >= 0 && best_d <= kGateDistance) {
        meas_to_track[mi] = best_ti;
        track_used[best_ti] = true;
      }
    }

    // 3) update matched
    for (std::size_t mi = 0; mi < measurements.size(); ++mi) {
      const int ti = meas_to_track[mi];
      if (ti < 0) {
        continue;
      }
      auto & tr = tracks_[static_cast<std::size_t>(ti)];
      tr.kf.update(measurements[mi].x(), measurements[mi].y(), kMeasurementNoise);
      tr.last_update_stamp = stamp;
      tr.missed = 0;

      result.filtered_xy[mi] = Eigen::Vector2d(tr.kf.px(), tr.kf.py());
      result.track_ids[mi] = tr.id;
    }

    // 4) create tracks for unmatched measurements
    for (std::size_t mi = 0; mi < measurements.size(); ++mi) {
      if (meas_to_track[mi] >= 0) {
        continue;
      }
      const int new_id = createTrack(measurements[mi].x(), measurements[mi].y(), stamp);
      const auto & tr = tracks_.back();
      result.filtered_xy[mi] = Eigen::Vector2d(tr.kf.px(), tr.kf.py());
      result.track_ids[mi] = new_id;
    }

    // 5) age / remove
    for (auto & tr : tracks_) {
      // 未匹配且本帧也没更新的 track，missed++
      if (tr.last_update_stamp != stamp) {
        tr.missed += 1;
      }
    }

    tracks_.erase(
      std::remove_if(
        tracks_.begin(), tracks_.end(),
        [&](const Track & tr) {
          return tr.missed > kMaxMissed;
        }),
      tracks_.end());

    return result;
  }

  std::string nextPcdFilePath()
  {
    namespace fs = std::filesystem;

    if (!pcd_list_initialized_) {
      pcd_list_initialized_ = true;
      pcd_files_.clear();
      pcd_index_ = 0;

      std::error_code ec;
      const fs::path input_path(pcd_path_);
      if (fs::is_directory(input_path, ec)) {
        for (const auto & entry : fs::directory_iterator(input_path, ec)) {
          if (ec) {
            break;
          }
          if (!entry.is_regular_file(ec)) {
            continue;
          }
          const auto & p = entry.path();
          if (p.has_extension() && p.extension() == ".pcd") {
            pcd_files_.push_back(p.string());
          }
        }
        std::sort(pcd_files_.begin(), pcd_files_.end());
      } else {
        pcd_files_.push_back(pcd_path_);
      }

      if (pcd_files_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "pcd_path 是目录但未找到任何 .pcd 文件: %s", pcd_path_.c_str());
        return {};
      }
    }

    if (pcd_files_.empty()) {
      return {};
    }

    if (pcd_index_ >= pcd_files_.size()) {
      pcd_index_ = 0;
    }
    return pcd_files_[pcd_index_++];
  }

  void computeAndPublish()
  {  
    // 加载点云数据（pcd_path 可以是单个文件，也可以是包含多个 .pcd 的目录）
    const std::string pcd_file = nextPcdFilePath();
    if (pcd_file.empty()) {
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) != 0) {
      RCLCPP_ERROR(this->get_logger(), "无法读取 PCD 文件: %s", pcd_file.c_str());
      return;
    }
    std::cout << "PointCloud before filtering has: " << cloud->size () << " data points." << std::endl; // 输出点云数量（滤波前）

    // 裁剪：用一个立方体范围 [crop_min, crop_max] 保留目标空间
    if (enable_crop_box_) 
    {
      pcl::CropBox<pcl::PointXYZ> crop;
      crop.setInputCloud(cloud);
      crop.setMin(Eigen::Vector4f(
        static_cast<float>(crop_min_[0]), static_cast<float>(crop_min_[1]), static_cast<float>(crop_min_[2]), 1.0f));
      crop.setMax(Eigen::Vector4f(
        static_cast<float>(crop_max_[0]), static_cast<float>(crop_max_[1]), static_cast<float>(crop_max_[2]), 1.0f));

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped(new pcl::PointCloud<pcl::PointXYZ>);
      crop.filter(*cloud_cropped);
      cloud = cloud_cropped;
      std::cout << "PointCloud after crop has: " << cloud->size () << " data points." << std::endl;
    }

    // 发布“后续处理所用的输入点云”（裁剪后）
    publishCloud(*cloud, cloud_pub_);

    // 体素滤波下采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    const float leaf = static_cast<float>(voxel_leaf_size_);
    vg.setLeafSize(leaf, leaf, leaf);
    vg.filter(*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->size ()  << " data points." << std::endl; // 输出点云数量（滤波后）

    // 平面分割并迭代移除最大平面
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rest(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(plane_max_iterations_);
    seg.setDistanceThreshold(static_cast<float>(plane_distance_threshold_));

    const int nr_points = static_cast<int>(cloud_filtered->size());
    // 迭代分割并移除平面，直到剩余点云数量小于原始数量的 30%
    while (cloud_filtered->size() > 0.3 * nr_points) 
    {
      // 从剩余点云中分割出最大的平面成分
      seg.setInputCloud(cloud_filtered);
      seg.segment(*inliers, *coefficients);
      if (inliers->indices.empty()) {
        break;
      }

      // 从输入点云中提取平面内点（inliers）
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(cloud_filtered);
      extract.setIndices(inliers);
      extract.setNegative(false);
      // 获取属于平面表面的点
      extract.filter(*cloud_plane);
      // 移除平面内点，提取剩余点云
      extract.setNegative(true);
      extract.filter(*cloud_rest);
      *cloud_filtered = *cloud_rest;
    }

    if (publish_filtered_cloud_ && filtered_cloud_pub_) {
      publishCloud(*cloud_filtered, filtered_cloud_pub_);
    }

    // 聚类
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>); // 为聚类提取创建用于搜索的 KdTree 对象
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(static_cast<float>(cluster_tolerance_));
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

    // 可视化聚类结果为立方体框
    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.reserve(cluster_indices.size());

    rclcpp::Time stamp = this->now();

    // 收集被标记为 robot 的观测中心点 (x,y)，用于 Kalman 滤波
    std::vector<Eigen::Vector2d> robot_measurements;
    // 每个 robot measurement 对应的文本 marker 在 marker_array 中的下标（用于回填 track_id 标签）
    std::vector<std::size_t> robot_label_marker_indices;

    // 先清空上一次的 marker，避免聚类数量变少时旧框残留
    {
      visualization_msgs::msg::Marker clear;
      clear.header.frame_id = frame_id_;
      clear.header.stamp = stamp;
      clear.action = visualization_msgs::msg::Marker::DELETEALL;
      marker_array.markers.push_back(std::move(clear));
    }

    int id = 0;
    int robot_num = 0;
    for (const auto & cluster : cluster_indices) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
      cloud_cluster->reserve(cluster.indices.size());
      for (const auto idx : cluster.indices) {
        cloud_cluster->push_back((*cloud_filtered)[idx]);
      }

      pcl::PointXYZ min_pt;
      pcl::PointXYZ max_pt;
      pcl::getMinMax3D(*cloud_cluster, min_pt, max_pt);

      const float cx = 0.5f * (min_pt.x + max_pt.x);
      const float cy = 0.5f * (min_pt.y + max_pt.y);
      const float cz = 0.5f * (min_pt.z + max_pt.z);

      const float sx = std::max(0.001f, max_pt.x - min_pt.x);
      const float sy = std::max(0.001f, max_pt.y - min_pt.y);
      const float sz = std::max(0.001f, max_pt.z - min_pt.z);

      // 判断聚类框的尺寸是否在合理范围内
      constexpr float kRobotMaxSizeM = 1.5f,
                      kRobotMinSizeM = 0.15f;
      const bool is_robot_box = (sx < kRobotMaxSizeM) && (sy < kRobotMaxSizeM) && (sz < kRobotMaxSizeM) &&
                                (sx > kRobotMinSizeM) && (sz > kRobotMinSizeM);

      if (is_robot_box) {
        robot_measurements.emplace_back(static_cast<double>(cx), static_cast<double>(cy));
      }

      visualization_msgs::msg::Marker m;
      m.header.frame_id = frame_id_;
      m.header.stamp = stamp;
      m.ns = "cluster_boxes";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;

      m.pose.position.x = cx;
      m.pose.position.y = cy;
      m.pose.position.z = cz;
      m.pose.orientation.w = 1.0;

      m.scale.x = sx;
      m.scale.y = sy;
      m.scale.z = sz;

      const float hue = 360.0f * (static_cast<float>(m.id) / std::max(1.0f, static_cast<float>(cluster_indices.size())));
      const auto rgb = hsvToRgb(hue, 0.9f, 0.9f);
      m.color.r = rgb.r;
      m.color.g = rgb.g;
      m.color.b = rgb.b;
      m.color.a = 0.35f;

      m.lifetime = rclcpp::Duration(0, 0);

      marker_array.markers.push_back(std::move(m));

      // 如果尺寸合理，认为是机器人并添加文本标签
      if (is_robot_box) {
        visualization_msgs::msg::Marker t;
        t.header.frame_id = frame_id_;
        t.header.stamp = stamp;
        t.ns = "robot_labels";
        t.id = id - 1;
        t.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        t.action = visualization_msgs::msg::Marker::ADD;

        t.pose.position.x = cx;
        t.pose.position.y = cy;
        t.pose.position.z = max_pt.z + 0.05f;
        t.pose.orientation.w = 1.0;

        t.scale.z = 0.2f;
        t.color.r = 1.0f;
        t.color.g = 1.0f;
        t.color.b = 1.0f;
        t.color.a = 0.9f;
        t.text = std::string("robot_") + std::to_string(robot_num++);

        t.lifetime = rclcpp::Duration(0, 0);
        robot_label_marker_indices.push_back(marker_array.markers.size());
        marker_array.markers.push_back(std::move(t));
      }
    }

    // Kalman 更新与记录（只针对 robot 观测）
    const auto track_result = updateTracks(robot_measurements, stamp);

    // 回填 RViz 文本标签：robot_<track_id>
    for (std::size_t i = 0; i < robot_label_marker_indices.size(); ++i) {
      const std::size_t mi = i;  // robot_label_marker_indices 与 robot_measurements 同步增长
      if (mi >= track_result.track_ids.size()) {
        continue;
      }
      const std::size_t marker_i = robot_label_marker_indices[i];
      if (marker_i >= marker_array.markers.size()) {
        continue;
      }
      const int tid = track_result.track_ids[mi];
      if (tid >= 0) {
        marker_array.markers[marker_i].text = std::string("robot_") + std::to_string(tid);
      }
    }

    // 同一帧多个 robot：并排拼成一行（label 与 RViz 保持一致：robot_<track_id>）
    std::string robots_inline;
    if (!robot_measurements.empty()) {
      std::ostringstream oss;
      oss.setf(std::ios::fixed);
      oss << std::setprecision(2);
      for (std::size_t i = 0; i < robot_measurements.size(); ++i) {
        if (i > 0) {
          oss << " | ";
        }
        const int tid = (i < track_result.track_ids.size()) ? track_result.track_ids[i] : -1;
        const auto & xy = (i < track_result.filtered_xy.size()) ? track_result.filtered_xy[i] : robot_measurements[i];
        if (tid >= 0) {
          oss << "robot_" << tid;
        } else {
          oss << "robot_" << i;
        }
        oss << "=(" << xy.x() << "," << xy.y() << ")";
      }
      robots_inline = oss.str();
    }

    if (record_stream_.is_open()) {
      // CSV 第三列用引号包住，避免里面的逗号/分隔符影响解析
      const int64_t stamp_ns = static_cast<int64_t>(stamp.nanoseconds());
      const int64_t stamp_sec = stamp_ns / 1000000000LL;
      const int64_t stamp_nanosec = stamp_ns % 1000000000LL;
      record_stream_ << stamp_sec << "," << stamp_nanosec << ",\"" << robots_inline << "\"\n";
      record_stream_.flush();
    }

    markers_pub_->publish(marker_array);

    RCLCPP_INFO(
      this->get_logger(),
      "发布点云：聚类框=%zu，robot标签=%u%s%s。PCD路径=%s",
      cluster_indices.size(),
      robot_num,
      robots_inline.empty() ? "" : "，",
      robots_inline.c_str(),
      pcd_file.c_str());
  }

  void publishCloud(
    const pcl::PointCloud<pcl::PointXYZ> & cloud,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub)
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = frame_id_;
    msg.header.stamp = this->now();
    pub->publish(msg);
  }

  std::string pcd_path_;
  std::string frame_id_;

  bool pcd_list_initialized_{false};
  std::vector<std::string> pcd_files_;
  std::size_t pcd_index_{0};

  double publish_rate_hz_{};

  double voxel_leaf_size_{};
  double plane_distance_threshold_{};
  int plane_max_iterations_{};

  double cluster_tolerance_{};
  int min_cluster_size_{};
  int max_cluster_size_{};

  bool enable_crop_box_{};
  std::vector<double> crop_min_{};
  std::vector<double> crop_max_{};

  bool publish_filtered_cloud_{};

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Kalman tracking + recording
  std::string start_time_str_;
  std::string record_path_;
  std::ofstream record_stream_;
  int next_track_id_{0};
  std::vector<Track> tracks_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ClusterRvizNode>());
  rclcpp::shutdown();
  return 0;
}
