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

#include <algorithm>
#include <filesystem>

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

    const auto period = std::chrono::duration<double>(1.0 / std::max(0.1, publish_rate_hz_));
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { this->computeAndPublish(); });
  }

private:
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
        marker_array.markers.push_back(std::move(t));
      }
    }

    markers_pub_->publish(marker_array);

    RCLCPP_INFO(
      this->get_logger(),
      "发布点云：聚类框=%zu，robot标签=%u。PCD路径=%s",
      cluster_indices.size(),
      robot_num,
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
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ClusterRvizNode>());
  rclcpp::shutdown();
  return 0;
}
