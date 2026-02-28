#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
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

#include <cmath>
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

    voxel_leaf_size_ = this->declare_parameter<double>("voxel_leaf_size", 0.01);
    plane_distance_threshold_ = this->declare_parameter<double>("plane_distance_threshold", 0.02);
    plane_max_iterations_ = this->declare_parameter<int>("plane_max_iterations", 100);

    cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.02);
    min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 100);
    max_cluster_size_ = this->declare_parameter<int>("max_cluster_size", 25000);

    publish_filtered_cloud_ = this->declare_parameter<bool>("publish_filtered_cloud", true);

    // PointCloud2 在 RViz 中通常使用 SensorData QoS（best effort）。
    const auto cloud_qos = rclcpp::SensorDataQoS();
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("input_cloud", cloud_qos);
    if (publish_filtered_cloud_) {
      filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_cloud", cloud_qos);
    }

    // MarkerArray 用默认 reliable QoS 即可。
    const auto marker_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("cluster_boxes", marker_qos);

    const auto period = std::chrono::duration<double>(1.0 / std::max(0.1, publish_rate_hz_));
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { this->computeAndPublish(); });
  }

private:
  void computeAndPublish()
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path_, *cloud) != 0) {
      RCLCPP_ERROR(this->get_logger(), "无法读取 PCD 文件: %s", pcd_path_.c_str());
      return;
    }

    publishCloud(*cloud, cloud_pub_);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    {
      pcl::VoxelGrid<pcl::PointXYZ> vg;
      vg.setInputCloud(cloud);
      const float leaf = static_cast<float>(voxel_leaf_size_);
      vg.setLeafSize(leaf, leaf, leaf);
      vg.filter(*cloud_filtered);
    }

    // 平面分割并迭代移除最大平面
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(plane_max_iterations_);
    seg.setDistanceThreshold(static_cast<float>(plane_distance_threshold_));

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rest(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    const int nr_points = static_cast<int>(cloud_filtered->size());
    while (cloud_filtered->size() > 0.3 * nr_points) {
      seg.setInputCloud(cloud_filtered);
      seg.segment(*inliers, *coefficients);
      if (inliers->indices.empty()) {
        break;
      }

      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(cloud_filtered);
      extract.setIndices(inliers);

      extract.setNegative(false);
      extract.filter(*cloud_plane);

      extract.setNegative(true);
      extract.filter(*cloud_rest);
      *cloud_filtered = *cloud_rest;
    }

    if (publish_filtered_cloud_ && filtered_cloud_pub_) {
      publishCloud(*cloud_filtered, filtered_cloud_pub_);
    }

    // 聚类
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(static_cast<float>(cluster_tolerance_));
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.reserve(cluster_indices.size());

    rclcpp::Time stamp = this->now();

    int id = 0;
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
    }

    markers_pub_->publish(marker_array);

    RCLCPP_INFO(
      this->get_logger(), "发布点云与 %zu 个聚类立方体框。PCD=%s", marker_array.markers.size(),
      pcd_path_.c_str());
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

  double publish_rate_hz_{};

  double voxel_leaf_size_{};
  double plane_distance_threshold_{};
  int plane_max_iterations_{};

  double cluster_tolerance_{};
  int min_cluster_size_{};
  int max_cluster_size_{};

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
