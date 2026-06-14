/**
 * 从PCD点云中分割最大垂直平面（平面平行于Z轴），并以该平面为基准构造坐标系，
 * 输出平面在该坐标系下的相对位姿，同时可视化平面框和坐标轴。
 *
 * 坐标系定义：
 *   - x 轴：平面平行方向（y轴绕Z轴旋转90度得到，保证在XY平面内）
 *   - y 轴：垂直平面方向（平面法向量在XY平面的投影）
 *   - z 轴：世界坐标系 Z 轴方向（向上）
 *   - 原点：平面点云在 x 方向上的最小位置
 *
 * Usage:
 *   plane_pose_estimation <input.pcd> [options]
 *   Options:
 *     --voxel <size>           体素下采样大小 (米)，0表示不下采样 (默认: 0)
 *     --distance <threshold>   RANSAC 距离阈值 (默认: 0.02)
 *     --iterations <n>         RANSAC 迭代次数 (默认: 1000)
 *     --x-min <val>            X 轴最小范围 (默认: -7.5)
 *     --x-max <val>            X 轴最大范围 (默认: 7.5)
 *     --y-min <val>            Y 轴最小范围 (默认: 0)
 *     --y-max <val>            Y 轴最大范围 (默认: 8)
 *     --z-min <val>            Z 轴最小范围 (默认: 0)
 *     --z-max <val>            Z 轴最大范围 (默认: 3)
 *     --out-plane <path>       保存平面点云 PCD 路径
 *     --out-cropped <path>     保存裁剪后点云 PCD 路径
 *     --no-viz                 关闭可视化窗口
 *     --axis-length <val>      坐标轴箭头长度 (默认: 1.0)
 */

#include <iostream>
#include <string>
#include <cmath>
#include <limits>
#include <cerrno>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// ========== 目标平面尺寸（毫米） ==========
// 期望的长（x方向，平面平行方向）和宽（z方向，高度），误差 ±200mm
static constexpr float kTargetLengthM = 6.130f;  // 6130mm
static constexpr float kTargetWidthM  = 0.770f;  // 770mm
static constexpr float kSizeToleranceM = 0.200f;  // ±200mm

// 网左侧在地图坐标系下的坐标
static constexpr float kNetLeftX = 0.58f; // 米
static constexpr float kNetLeftY = 7.85f;  // 米

// ========== 参数解析辅助函数 ==========

static bool tryParseFloat(const char* s, float& out)
{
  if (s == nullptr || *s == '\0')
    return false;
  char* end = nullptr;
  errno = 0;
  float v = std::strtof(s, &end);
  if (errno != 0 || end == s || *end != '\0')
    return false;
  out = v;
  return true;
}

static bool tryParseInt(const char* s, int& out)
{
  if (s == nullptr || *s == '\0')
    return false;
  char* end = nullptr;
  errno = 0;
  long v = std::strtol(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0')
    return false;
  if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
    return false;
  out = static_cast<int>(v);
  return true;
}

static bool hasOption(int argc, char* argv[], const std::string& option)
{
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == option)
      return true;
  return false;
}

static std::string getOption(int argc, char* argv[], const std::string& option, const std::string& default_val = "")
{
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == option)
      return std::string(argv[i + 1]);
  return default_val;
}

// ========== 裁剪点云 ==========

static void cropPointCloud(const PointCloudT::Ptr& cloud,
                           PointCloudT::Ptr& cloud_out,
                           float x_min, float x_max,
                           float y_min, float y_max,
                           float z_min, float z_max)
{
  cloud_out->clear();
  for (const auto& pt : cloud->points)
  {
    if (pt.x >= x_min && pt.x <= x_max &&
        pt.y >= y_min && pt.y <= y_max &&
        pt.z >= z_min && pt.z <= z_max)
    {
      cloud_out->push_back(pt);
    }
  }
  cloud_out->width = cloud_out->size();
  cloud_out->height = 1;
  cloud_out->is_dense = true;
}

// ========== 离群点过滤 ==========

// 在平面点云上做半径离群点移除：在平面局部坐标系中，
// 剔除 xz 平面内邻居过少的孤立点（排除平面上稀疏噪点）
static void removePlaneOutliers(const PointCloudT::Ptr& cloud_in,
                                PointCloudT::Ptr& cloud_out,
                                float radius = 0.1f,
                                int min_neighbors = 5)
{
  pcl::RadiusOutlierRemoval<PointT> ror;
  ror.setInputCloud(cloud_in);
  ror.setRadiusSearch(radius);
  ror.setMinNeighborsInRadius(min_neighbors);
  ror.filter(*cloud_out);

  std::cout << "    Outlier removal: " << cloud_in->size() << " -> " << cloud_out->size()
            << " (radius=" << radius << "m, min_neighbors=" << min_neighbors << ")" << std::endl;
}


// ========== 平面点云 bounding box ==========

struct PlaneBBox
{
  // 8 个角点（世界坐标系下）
  Eigen::Vector3f corners[8];
  // 平面坐标系下的范围
  float x_min, x_max, y_min, y_max, z_min, z_max;
};

// ========== 左侧墙面数据结构 ==========

struct LeftWall
{
  PointCloudT::Ptr plane_cloud;
  pcl::ModelCoefficients coefficients;
  PlaneBBox bbox;
  int index; // 墙面编号
};

// ========== 辅助：为垂直平面计算 PlaneBBox ==========

static bool computeBBoxForVerticalPlane(const PointCloudT::Ptr& cloud,
                                         const pcl::ModelCoefficients& coeffs,
                                         PlaneBBox& bbox)
{
  Eigen::Vector3f normal(coeffs.values[0], coeffs.values[1], coeffs.values[2]);
  normal.normalize();

  Eigen::Vector3f y_axis(normal.x(), normal.y(), 0.0f);
  float yn = y_axis.norm();
  if (yn < 1e-10f)
    return false;
  y_axis.normalize();
  Eigen::Vector3f x_axis(-y_axis.y(), y_axis.x(), 0.0f);
  Eigen::Vector3f z_axis(0, 0, 1);

  Eigen::Matrix3f R;
  R.col(0) = x_axis;
  R.col(1) = y_axis;
  R.col(2) = z_axis;

  bbox.x_min = bbox.y_min = bbox.z_min = std::numeric_limits<float>::max();
  bbox.x_max = bbox.y_max = bbox.z_max = -std::numeric_limits<float>::max();
  for (const auto& pt : cloud->points)
  {
    Eigen::Vector3f p_local = R.transpose() * Eigen::Vector3f(pt.x, pt.y, pt.z);
    if (p_local.x() < bbox.x_min) bbox.x_min = p_local.x();
    if (p_local.x() > bbox.x_max) bbox.x_max = p_local.x();
    if (p_local.y() < bbox.y_min) bbox.y_min = p_local.y();
    if (p_local.y() > bbox.y_max) bbox.y_max = p_local.y();
    if (p_local.z() < bbox.z_min) bbox.z_min = p_local.z();
    if (p_local.z() > bbox.z_max) bbox.z_max = p_local.z();
  }

  float lx[2] = {bbox.x_min, bbox.x_max};
  float ly[2] = {bbox.y_min, bbox.y_max};
  float lz[2] = {bbox.z_min, bbox.z_max};
  int idx = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        bbox.corners[idx++] = R * Eigen::Vector3f(lx[i], ly[j], lz[k]);

  return true;
}

// ========== 辅助：判断平面重心是否在 baselink 左侧 (x < 0) ==========

static bool isOnLeftSide(const PointCloudT::Ptr& cloud)
{
  float cx = 0.0f;
  for (const auto& pt : cloud->points)
    cx += pt.x;
  cx /= static_cast<float>(cloud->size());
  return cx < 0.0f;
}

// ========== 迭代分割平面，找到尺寸匹配目标的平面 ==========

// 返回值: 1=精确匹配, 0=未匹配但返回最佳候选, -1=无可用平面
// 同时收集左侧 (x<0) 的垂直墙面到 left_walls
static int segmentTargetPlane(const PointCloudT::Ptr& cloud,
                               PointCloudT::Ptr& plane_cloud,
                               pcl::ModelCoefficients& coefficients,
                               float distance_threshold,
                               int max_iterations,
                               std::vector<LeftWall>& left_walls,
                               bool filter_outliers = true,
                               float outlier_radius = 0.1f,
                               int outlier_min_neighbors = 5,
                               int max_planes = 20)
{
  if (cloud->size() < 200)
  {
    PCL_ERROR("Cropped cloud has too few points (%lu), cannot segment plane\n", cloud->size());
    return -1;
  }

  PointCloudT::Ptr cloud_rest(new PointCloudT);
  *cloud_rest = *cloud;

  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(max_iterations);
  seg.setDistanceThreshold(distance_threshold);

  const int nr_points = static_cast<int>(cloud_rest->size());

  // 记录最佳候选平面（误差最小的那个）
  PointCloudT::Ptr best_plane(new PointCloudT);
  pcl::ModelCoefficients best_coeffs;
  float best_score = std::numeric_limits<float>::max();
  bool has_best = false;

  int plane_idx = 0;
  while (cloud_rest->size() > 0.1f * nr_points && plane_idx < max_planes)
  {
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients coeffs;

    seg.setInputCloud(cloud_rest);
    seg.segment(*inliers, coeffs);

    if (inliers->indices.empty())
    {
      std::cout << "  Cannot estimate more plane models, stopping." << std::endl;
      break;
    }

    // 提取当前平面内点
    PointCloudT::Ptr current_plane_raw(new PointCloudT);
    {
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud(cloud_rest);
      extract.setIndices(inliers);
      extract.setNegative(false);
      extract.filter(*current_plane_raw);
    }

    // 离群点过滤：剔除平面上稀疏的孤立噪点
    PointCloudT::Ptr current_plane(new PointCloudT);
    if (filter_outliers && current_plane_raw->size() > 50)
    {
      removePlaneOutliers(current_plane_raw, current_plane, outlier_radius, outlier_min_neighbors);
    }
    else
    {
      current_plane = current_plane_raw;
    }

    // 检查平面法向量是否接近水平（平行于Z轴）
    Eigen::Vector3f normal(coeffs.values[0], coeffs.values[1], coeffs.values[2]);
    normal.normalize();
    float z_comp = std::abs(normal.z());

    // 计算该平面在平面坐标系下的尺寸（用于筛选）
    // 构造临时坐标系计算 bbox
    Eigen::Vector3f y_axis = normal;
    y_axis.z() = 0.0f;
    float y_norm = y_axis.norm();
    if (y_norm < 1e-10f)
    {
      // 法向量完全垂直，不适合作为垂直平面
      std::cout << "  Plane " << plane_idx << ": normal is perfectly vertical, skipping (z_comp=" << z_comp << ")" << std::endl;
      // 移除该平面，继续找下一个
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud(cloud_rest);
      extract.setIndices(inliers);
      extract.setNegative(true);
      PointCloudT::Ptr tmp(new PointCloudT);
      extract.filter(*tmp);
      *cloud_rest = *tmp;
      ++plane_idx;
      continue;
    }
    y_axis.normalize();
    Eigen::Vector3f x_axis(-y_axis.y(), y_axis.x(), 0.0f);
    Eigen::Matrix3f R;
    R.col(0) = x_axis;
    R.col(1) = y_axis;
    R.col(2) = Eigen::Vector3f(0, 0, 1);

    // 计算 bbox
    float x_min_l = std::numeric_limits<float>::max();
    float x_max_l = -std::numeric_limits<float>::max();
    float z_min_l = std::numeric_limits<float>::max();
    float z_max_l = -std::numeric_limits<float>::max();
    for (const auto& pt : current_plane->points)
    {
      Eigen::Vector3f p_local = R.transpose() * Eigen::Vector3f(pt.x, pt.y, pt.z);
      if (p_local.x() < x_min_l) x_min_l = p_local.x();
      if (p_local.x() > x_max_l) x_max_l = p_local.x();
      if (p_local.z() < z_min_l) z_min_l = p_local.z();
      if (p_local.z() > z_max_l) z_max_l = p_local.z();
    }
    float length = x_max_l - x_min_l;
    float width  = z_max_l - z_min_l;
    float length_err = std::abs(length - kTargetLengthM);
    float width_err  = std::abs(width  - kTargetWidthM);

    bool length_ok = length_err <= kSizeToleranceM;
    bool width_ok  = width_err  <= kSizeToleranceM;

    std::cout << "  Plane " << plane_idx
              << ": points=" << current_plane->size()
              << ", z_comp=" << z_comp
              << ", L(x)=" << length * 1000.0f << "mm (target 6130±200mm)"
              << ", W(z)=" << width * 1000.0f << "mm (target 770±200mm)";

    if (length_ok && width_ok)
    {
      std::cout << " [MATCHED!]" << std::endl;
      *plane_cloud = *current_plane;
      coefficients = coeffs;
      return 1;
    }
    else
    {
      std::cout << " [no match]" << std::endl;
    }

    // 记录最佳候选
    float score = length_err + width_err;
    if (score < best_score)
    {
      best_score = score;
      *best_plane = *current_plane;
      best_coeffs = coeffs;
      has_best = true;
    }

    // 如果该平面位于 baselink 左侧 (x < 0)，收集为左侧墙面
    if (isOnLeftSide(current_plane))
    {
      PlaneBBox wall_bbox;
      if (computeBBoxForVerticalPlane(current_plane, coeffs, wall_bbox))
      {
        LeftWall wall;
        wall.plane_cloud = current_plane;
        wall.coefficients = coeffs;
        wall.bbox = wall_bbox;
        wall.index = static_cast<int>(left_walls.size());
        left_walls.push_back(wall);

        float w_len = wall_bbox.x_max - wall_bbox.x_min;
        float w_hgt = wall_bbox.z_max - wall_bbox.z_min;
        std::cout << "    [left wall #" << wall.index
                  << "] length(x)=" << w_len << "m"
                  << ", height(z)=" << w_hgt << "m" << std::endl;
      }
    }

    // 移除当前平面，继续找下一个
    {
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud(cloud_rest);
      extract.setIndices(inliers);
      extract.setNegative(true);
      PointCloudT::Ptr tmp(new PointCloudT);
      extract.filter(*tmp);
      *cloud_rest = *tmp;
    }
    ++plane_idx;
  }

  if (has_best)
  {
    PCL_ERROR("No exact match in %d planes (target: L=6130±200mm, W=770±200mm)\n", plane_idx);
    PCL_ERROR("Using best candidate (smallest combined error)\n");
    *plane_cloud = *best_plane;
    coefficients = best_coeffs;
    return 0;
  }

  PCL_ERROR("No valid plane found\n");
  return -1;
}

// ========== 计算平面坐标系位姿 ==========

struct PlanePose
{
  Eigen::Vector3f position;      // 平移向量 (世界坐标系下平面坐标系原点的位置)
  Eigen::Matrix3f rotation;      // 旋转矩阵 (平面坐标系 -> 世界坐标系)
  Eigen::Vector3f x_axis;        // 平面平行方向
  Eigen::Vector3f y_axis;        // 垂直平面方向
  Eigen::Vector3f z_axis;        // 向上方向
  Eigen::Vector3f plane_normal;  // 原始平面法向量
};

// ========== baselink -> net TF 变换 ==========
// 输入: baselink 坐标系下的平面点云
// 输出: net 坐标系原点 = 平面左下角角点 (x_min 方向的端点)
//       net 坐标系 x 轴 = Xmin->Xmax 方向 (平面平行方向)
//       net 坐标系 z 轴 = 世界 Z 轴
//       平移量 = (左下角x, 左下角y, 0)

static bool computePlanePose(const PointCloudT::Ptr& plane_cloud,
                             const pcl::ModelCoefficients& coefficients,
                             PlanePose& pose)
{
  // 平面法向量
  Eigen::Vector3f normal(coefficients.values[0],
                          coefficients.values[1],
                          coefficients.values[2]);
  float norm = normal.norm();
  if (norm < 1e-10f)
  {
    PCL_ERROR("Plane normal has near-zero magnitude\n");
    return false;
  }
  normal.normalize();

  std::cout << "  Plane normal: [" << normal.x() << ", " << normal.y() << ", " << normal.z() << "]" << std::endl;

  // 平面平行方向：法向量在XY平面的投影，绕Z轴旋转+90度得到
  // 即 (-ny, nx, 0)，这就是 Xmin->Xmax 的方向
  Eigen::Vector3f plane_dir(-normal.y(), normal.x(), 0.0f);
  float dir_norm = plane_dir.norm();
  if (dir_norm < 1e-10f)
  {
    PCL_ERROR("Cannot determine plane parallel direction (normal is near vertical)\n");
    return false;
  }
  plane_dir.normalize();

  // 将平面点云投影到 plane_dir 方向上，找出 x_min 和 x_max 的端点
  // 同时记录对应的 y 值，用于确定左下角角点
  float x_min = std::numeric_limits<float>::max();
  float x_max = -std::numeric_limits<float>::max();
  Eigen::Vector3f p_xmin, p_xmax;  // x_min 和 x_max 对应的3D点
  float z_min = std::numeric_limits<float>::max();
  float z_max = -std::numeric_limits<float>::max();

  for (const auto& pt : plane_cloud->points)
  {
    Eigen::Vector3f p(pt.x, pt.y, pt.z);
    float proj = p.dot(plane_dir);
    if (proj < x_min) { x_min = proj; p_xmin = p; }
    if (proj > x_max) { x_max = proj; p_xmax = p; }
    if (p.z() < z_min) z_min = p.z();
    if (p.z() > z_max) z_max = p.z();
  }

  // 左下角角点：x_min 端点的 (x, y) 坐标，z 取最低点
  // 平移量 = (左下角x, 左下角y, 0)
  Eigen::Vector3f translation(p_xmin.x(), p_xmin.y(), 0.0f);

  // 旋转：Xmin->Xmax 方向作为 net 坐标系的 x 轴
  Eigen::Vector3f x_axis = plane_dir;
  // y 轴 = z_axis × x_axis (右手定则)
  Eigen::Vector3f z_axis(0.0f, 0.0f, 1.0f);
  Eigen::Vector3f y_axis = z_axis.cross(x_axis);
  y_axis.normalize();

  // 旋转矩阵: baselink -> net, 列向量 = x, y, z
  Eigen::Matrix3f R;
  R.col(0) = x_axis;
  R.col(1) = y_axis;
  R.col(2) = z_axis;

  // 输出
  pose.position = translation;
  pose.rotation = R;
  pose.x_axis = x_axis;
  pose.y_axis = y_axis;
  pose.z_axis = z_axis;
  pose.plane_normal = normal;

  // 计算 yaw 角度（绕 Z 轴的旋转角）
  float yaw = std::atan2(x_axis.y(), x_axis.x());

  std::cout << "  Plane dir (Xmin->Xmax): [" << x_axis.x() << ", " << x_axis.y() << ", " << x_axis.z() << "]"
            << ", yaw=" << yaw * 180.0f / M_PI << "deg" << std::endl;
  std::cout << "  Bottom-left corner (net origin): [" << translation.x() << ", " << translation.y() << ", " << translation.z() << "]"
            << std::endl;
  std::cout << "  Plane size: L(x)=" << (x_max - x_min) << "m, H(z)=" << (z_max - z_min) << "m" << std::endl;

  return true;
}

// ========== 计算平面点云的 bounding box 角点（在平面坐标系下）==========

static PlaneBBox computePlaneBBox(const PointCloudT::Ptr& plane_cloud,
                                  const PlanePose& pose)
{
  PlaneBBox bbox;
  const Eigen::Matrix3f& R = pose.rotation;

  bbox.x_min = bbox.y_min = bbox.z_min = std::numeric_limits<float>::max();
  bbox.x_max = bbox.y_max = bbox.z_max = -std::numeric_limits<float>::max();

  for (const auto& pt : plane_cloud->points)
  {
    Eigen::Vector3f p_world(pt.x, pt.y, pt.z);
    Eigen::Vector3f p_local = R.transpose() * p_world;
    if (p_local.x() < bbox.x_min) bbox.x_min = p_local.x();
    if (p_local.x() > bbox.x_max) bbox.x_max = p_local.x();
    if (p_local.y() < bbox.y_min) bbox.y_min = p_local.y();
    if (p_local.y() > bbox.y_max) bbox.y_max = p_local.y();
    if (p_local.z() < bbox.z_min) bbox.z_min = p_local.z();
    if (p_local.z() > bbox.z_max) bbox.z_max = p_local.z();
  }

  // 在平面坐标系下构造 8 个角点，再变换到世界坐标系
  float lx[2] = {bbox.x_min, bbox.x_max};
  float ly[2] = {bbox.y_min, bbox.y_max};
  float lz[2] = {bbox.z_min, bbox.z_max};
  int idx = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
      {
        Eigen::Vector3f corner_local(lx[i], ly[j], lz[k]);
        bbox.corners[idx++] = R * corner_local;
      }

  return bbox;
}

// ========== 打印结果 ==========

static void printPose(const PlanePose& pose)
{
  float yaw = std::atan2(pose.x_axis.y(), pose.x_axis.x());

  printf("\n  === baselink -> net TF ===\n");
  printf("  Translation (x, y, z):\n");
  printf("    x = %8.6f, y = %8.6f, z = %8.6f\n",
         pose.position.x(), pose.position.y(), pose.position.z());
  printf("  Rotation (yaw):\n");
  printf("    yaw = %8.6f rad (%8.3f deg)\n", yaw, yaw * 180.0f / M_PI);
  printf("  net x_axis (Xmin->Xmax): [%8.6f, %8.6f, %8.6f]\n",
         pose.x_axis.x(), pose.x_axis.y(), pose.x_axis.z());
  printf("  net y_axis:              [%8.6f, %8.6f, %8.6f]\n",
         pose.y_axis.x(), pose.y_axis.y(), pose.y_axis.z());
  printf("  net z_axis:              [%8.6f, %8.6f, %8.6f]\n",
         pose.z_axis.x(), pose.z_axis.y(), pose.z_axis.z());
  printf("  Rotation matrix (net -> baselink):\n");
  printf("    [%8.6f, %8.6f, %8.6f]\n",
         pose.rotation(0, 0), pose.rotation(0, 1), pose.rotation(0, 2));
  printf("    [%8.6f, %8.6f, %8.6f]\n",
         pose.rotation(1, 0), pose.rotation(1, 1), pose.rotation(1, 2));
  printf("    [%8.6f, %8.6f, %8.6f]\n",
         pose.rotation(2, 0), pose.rotation(2, 1), pose.rotation(2, 2));
}

// ========== 可视化 ==========

static void addCoordinateAxes(pcl::visualization::PCLVisualizer& viewer,
                              const Eigen::Vector3f& origin,
                              const Eigen::Vector3f& x_axis,
                              const Eigen::Vector3f& y_axis,
                              const Eigen::Vector3f& z_axis,
                              float length,
                              const std::string& prefix)
{
  // x 轴 - 红色
  {
    pcl::PointXYZ p1(origin.x(), origin.y(), origin.z());
    pcl::PointXYZ p2(origin.x() + x_axis.x() * length,
                     origin.y() + x_axis.y() * length,
                     origin.z() + x_axis.z() * length);
    std::string id = prefix + "_x_axis";
    viewer.addArrow(p2, p1, 1.0, 0.0, 0.0, false, id);
    viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, id);
  }

  // y 轴 - 绿色
  {
    pcl::PointXYZ p1(origin.x(), origin.y(), origin.z());
    pcl::PointXYZ p2(origin.x() + y_axis.x() * length,
                     origin.y() + y_axis.y() * length,
                     origin.z() + y_axis.z() * length);
    std::string id = prefix + "_y_axis";
    viewer.addArrow(p2, p1, 0.0, 1.0, 0.0, false, id);
    viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, id);
  }

  // z 轴 - 蓝色
  {
    pcl::PointXYZ p1(origin.x(), origin.y(), origin.z());
    pcl::PointXYZ p2(origin.x() + z_axis.x() * length,
                     origin.y() + z_axis.y() * length,
                     origin.z() + z_axis.z() * length);
    std::string id = prefix + "_z_axis";
    viewer.addArrow(p2, p1, 0.0, 0.0, 1.0, false, id);
    viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, id);
  }
}

static void addBoundingBox(pcl::visualization::PCLVisualizer& viewer,
                           const PlaneBBox& bbox,
                           const std::string& prefix)
{
  // 12 条棱的连接索引
  const int edges[12][2] = {
    {0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 5},
    {2, 3}, {2, 6}, {3, 7}, {4, 5}, {4, 6},
    {5, 7}, {6, 7}
  };

  for (int i = 0; i < 12; ++i)
  {
    int i0 = edges[i][0];
    int i1 = edges[i][1];
    const auto& c0 = bbox.corners[i0];
    const auto& c1 = bbox.corners[i1];

    pcl::PointXYZ p0(c0.x(), c0.y(), c0.z());
    pcl::PointXYZ p1(c1.x(), c1.y(), c1.z());

    std::string id = prefix + "_edge_" + std::to_string(i);
    viewer.addLine(p0, p1, 1.0, 1.0, 0.0, id); // 黄色框
    viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, id);
  }
}

static void visualize(const PointCloudT::Ptr& cloud_original,
                      const PointCloudT::Ptr& cloud_cropped,
                      const PointCloudT::Ptr& cloud_plane,
                      const PlanePose& pose,
                      const PlaneBBox& bbox,
                      float axis_length,
                      bool exact_match,
                      const Eigen::Vector3f& map_position,
                      const Eigen::Matrix3f& map_rotation,
                      const std::vector<LeftWall>& left_walls)
{
  std::string title = exact_match ? "Plane Pose Estimation [MATCHED]"
                                  : "Plane Pose Estimation [BEST CANDIDATE - no exact match]";
  pcl::visualization::PCLVisualizer viewer(title);

  // 背景色
  viewer.setBackgroundColor(0.05, 0.05, 0.05);

  // 1) Original cloud (within crop range) - white
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cropped_color(cloud_cropped, 200, 200, 200);
  viewer.addPointCloud<PointT>(cloud_cropped, cropped_color, "cloud_cropped");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_cropped");

  // 2) Segmented plane cloud - cyan (highlighted)
  pcl::visualization::PointCloudColorHandlerCustom<PointT> plane_color(cloud_plane, 0, 200, 255);
  viewer.addPointCloud<PointT>(cloud_plane, plane_color, "cloud_plane");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_plane");

  // 3) Plane bounding box - yellow wireframe
  addBoundingBox(viewer, bbox, "plane_bbox");

  // 3.5) Left wall point clouds (magenta) and bounding boxes (magenta wireframe)
  for (const auto& wall : left_walls)
  {
    std::string wall_prefix = "left_wall_" + std::to_string(wall.index);

    // 墙面点云 - 紫色
    pcl::visualization::PointCloudColorHandlerCustom<PointT> wall_color(wall.plane_cloud, 200, 50, 200);
    viewer.addPointCloud<PointT>(wall.plane_cloud, wall_color, wall_prefix + "_cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, wall_prefix + "_cloud");

    // 墙面 bounding box - 品红色线框
    const auto& wbbox = wall.bbox;
    const int edges[12][2] = {
      {0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 5},
      {2, 3}, {2, 6}, {3, 7}, {4, 5}, {4, 6},
      {5, 7}, {6, 7}
    };
    for (int e = 0; e < 12; ++e)
    {
      const auto& c0 = wbbox.corners[edges[e][0]];
      const auto& c1 = wbbox.corners[edges[e][1]];
      pcl::PointXYZ p0(c0.x(), c0.y(), c0.z());
      pcl::PointXYZ p1(c1.x(), c1.y(), c1.z());
      std::string eid = wall_prefix + "_edge_" + std::to_string(e);
      viewer.addLine(p0, p1, 1.0, 0.0, 1.0, eid); // 品红色
      viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, eid);
    }

    // 墙面编号标签（在 bbox 中心偏上）
    Eigen::Vector3f center(
      (wbbox.corners[0].x() + wbbox.corners[7].x()) * 0.5f,
      (wbbox.corners[0].y() + wbbox.corners[7].y()) * 0.5f,
      (wbbox.corners[0].z() + wbbox.corners[7].z()) * 0.5f);
    std::string label = "W" + std::to_string(wall.index);
    viewer.addText3D(label, pcl::PointXYZ(center.x(), center.y(), center.z() + 0.2f),
                     0.2, 1.0, 0.0, 1.0, wall_prefix + "_label");
  }

  // 4) net coordinate system (at net origin) - Red(X: Xmin->Xmax) Green(Y) Blue(Z: up)
  addCoordinateAxes(viewer, pose.position, pose.x_axis, pose.y_axis, pose.z_axis,
                    axis_length, "net");

  // 5) baselink coordinate system (at world origin, shorter axes for reference)
  {
    Eigen::Vector3f world_origin(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f wx(1, 0, 0), wy(0, 1, 0), wz(0, 0, 1);
    addCoordinateAxes(viewer, world_origin, wx, wy, wz, axis_length * 0.5f, "baselink");
  }

  // 5.5) map coordinate system (at baselink position in map frame)
  {
    Eigen::Vector3f mx = map_rotation.col(0);
    Eigen::Vector3f my = map_rotation.col(1);
    Eigen::Vector3f mz = map_rotation.col(2);
    addCoordinateAxes(viewer, map_position, mx, my, mz, axis_length * 0.6f, "map");
  }

  // 6) Add text labels
  viewer.addText3D("net X (Xmin->Xmax)", pcl::PointXYZ(
    pose.position.x() + pose.x_axis.x() * (axis_length + 0.2f),
    pose.position.y() + pose.x_axis.y() * (axis_length + 0.2f),
    pose.position.z() + pose.x_axis.z() * (axis_length + 0.2f)),
    0.15, 1.0, 0.0, 0.0, "label_x");
  viewer.addText3D("net Y", pcl::PointXYZ(
    pose.position.x() + pose.y_axis.x() * (axis_length + 0.2f),
    pose.position.y() + pose.y_axis.y() * (axis_length + 0.2f),
    pose.position.z() + pose.y_axis.z() * (axis_length + 0.2f)),
    0.15, 0.0, 1.0, 0.0, "label_y");
  viewer.addText3D("net Z (up)", pcl::PointXYZ(
    pose.position.x() + pose.z_axis.x() * (axis_length + 0.2f),
    pose.position.y() + pose.z_axis.y() * (axis_length + 0.2f),
    pose.position.z() + pose.z_axis.z() * (axis_length + 0.2f)),
    0.15, 0.0, 0.0, 1.0, "label_z");

  // 7) net origin sphere
  pcl::PointXYZ origin_pt(pose.position.x(), pose.position.y(), pose.position.z());
  viewer.addSphere(origin_pt, 0.1, 1.0, 0.8, 0.2, "origin_sphere");

  // 8) Add legend text
  viewer.addText("White: cropped cloud | Cyan: detected plane | Yellow box: plane bbox", 10, 20, 14,
                 1.0, 1.0, 1.0, "legend");
  viewer.addText("Red=X(Xmin->Xmax) Green=Y Blue=Z(up) | Half-length axes=baselink | 0.6x axes=map frame", 10, 45, 14,
                 1.0, 1.0, 1.0, "legend2");
  viewer.addText("Magenta: left wall (x<0) | Magenta box: wall bbox", 10, 70, 14,
                 1.0, 1.0, 1.0, "legend3");

  // Set initial viewpoint
  viewer.setCameraPosition(0.0, -8.0, 5.0,  0.0, 0.0, 1.0,  0.0, -1.0, 0.0);
  viewer.setSize(1280, 720);

  std::cout << "\nVisualization window opened. Press 'q' or close the window to exit." << std::endl;

  while (!viewer.wasStopped())
  {
    viewer.spinOnce(100);
  }
}

// ========== 主函数 ==========

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);

  // 默认参数
  float voxel_size = 0.01f;
  float distance_threshold = 0.02f;
  int ransac_iterations = 1000;
  float x_min = -7.5f, x_max = 7.5f;
  float y_min = 0.0f,  y_max = 8.0f;
  float z_min = 0.0f,  z_max = 3.0f;
  float axis_length = 1.0f;
  bool no_viz = false;
  bool filter_outliers = true;      // 默认开启离群点过滤
  float outlier_radius = 0.1f;      // 离群点过滤半径(米)
  int outlier_min_neighbors = 5;    // 半径内最少邻居数
  std::string out_plane_path;
  std::string out_cropped_path;

  // 第一个非选项参数为输入 PCD 路径
  std::string input_path;
  for (int i = 1; i < argc; ++i)
  {
    std::string arg(argv[i]);
    if (arg.rfind("--", 0) == 0)
    {
      // 跳过选项及其值
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
        ++i;
    }
    else
    {
      input_path = arg;
      break;
    }
  }

  if (input_path.empty())
  {
    printf("Usage: %s <input.pcd> [options]\n", argv[0]);
    printf("Options:\n");
    printf("  --voxel <size>           Voxel downsampling size (m), 0=disabled (default: 0)\n");
    printf("  --distance <threshold>   RANSAC distance threshold (default: 0.02)\n");
    printf("  --iterations <n>         RANSAC iterations (default: 1000)\n");
    printf("  --x-min <val>            X min crop range (default: -7.5)\n");
    printf("  --x-max <val>            X max crop range (default: 7.5)\n");
    printf("  --y-min <val>            Y min crop range (default: 0)\n");
    printf("  --y-max <val>            Y max crop range (default: 8)\n");
    printf("  --z-min <val>            Z min crop range (default: 0)\n");
    printf("  --z-max <val>            Z max crop range (default: 3)\n");
    printf("  --out-plane <path>       Save plane cloud to PCD file\n");
    printf("  --out-cropped <path>     Save cropped cloud to PCD file\n");
    printf("  --no-viz                 Disable visualization window\n");
    printf("  --no-outlier-filter      Disable outlier removal (enabled by default)\n");
    printf("  --outlier-radius <val>   Outlier removal search radius (m) (default: 0.1)\n");
    printf("  --outlier-min-neighbors <n>  Min neighbors in radius (default: 5)\n");
    printf("  --axis-length <val>      Coordinate axis arrow length (default: 1.0)\n");
    return -1;
  }

  // 解析选项
  no_viz = hasOption(argc, argv, "--no-viz");
  filter_outliers = !hasOption(argc, argv, "--no-outlier-filter");
  {
    std::string val;
    val = getOption(argc, argv, "--voxel");        if (!val.empty()) tryParseFloat(val.c_str(), voxel_size);
    val = getOption(argc, argv, "--distance");      if (!val.empty()) tryParseFloat(val.c_str(), distance_threshold);
    val = getOption(argc, argv, "--iterations");    { int v; if (!val.empty() && tryParseInt(val.c_str(), v)) ransac_iterations = v; }
    val = getOption(argc, argv, "--x-min");         if (!val.empty()) tryParseFloat(val.c_str(), x_min);
    val = getOption(argc, argv, "--x-max");         if (!val.empty()) tryParseFloat(val.c_str(), x_max);
    val = getOption(argc, argv, "--y-min");         if (!val.empty()) tryParseFloat(val.c_str(), y_min);
    val = getOption(argc, argv, "--y-max");         if (!val.empty()) tryParseFloat(val.c_str(), y_max);
    val = getOption(argc, argv, "--z-min");         if (!val.empty()) tryParseFloat(val.c_str(), z_min);
    val = getOption(argc, argv, "--z-max");         if (!val.empty()) tryParseFloat(val.c_str(), z_max);
    val = getOption(argc, argv, "--axis-length");   if (!val.empty()) tryParseFloat(val.c_str(), axis_length);
    val = getOption(argc, argv, "--outlier-radius"); if (!val.empty()) tryParseFloat(val.c_str(), outlier_radius);
    { int v; val = getOption(argc, argv, "--outlier-min-neighbors"); if (!val.empty() && tryParseInt(val.c_str(), v)) outlier_min_neighbors = v; }
    out_plane_path   = getOption(argc, argv, "--out-plane");
    out_cropped_path = getOption(argc, argv, "--out-cropped");
  }

  // ===== 1. Load point cloud =====
  std::cout << "Loading point cloud: " << input_path << std::endl;
  PointCloudT::Ptr cloud(new PointCloudT);
  if (pcl::io::loadPCDFile<PointT>(input_path, *cloud) < 0)
  {
    PCL_ERROR("Failed to load PCD file: %s\n", input_path.c_str());
    return -1;
  }
  std::cout << "  Original point count: " << cloud->size() << std::endl;

  // ===== 2. Voxel downsampling (optional) =====
  PointCloudT::Ptr cloud_processed(new PointCloudT);
  if (voxel_size > 0.0f)
  {
    std::cout << "Voxel downsampling (voxel_size=" << voxel_size << "m)..." << std::endl;
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    vg.filter(*cloud_processed);
    std::cout << "  Downsampled point count: " << cloud_processed->size() << std::endl;
  }
  else
  {
    cloud_processed = cloud;
  }

  // ===== 3. Crop observation range =====
  std::cout << "\nCropping: x=[" << x_min << ", " << x_max
            << "], y=[" << y_min << ", " << y_max
            << "], z=[" << z_min << ", " << z_max << "]" << std::endl;

  PointCloudT::Ptr cloud_cropped(new PointCloudT);
  cropPointCloud(cloud_processed, cloud_cropped, x_min, x_max, y_min, y_max, z_min, z_max);
  std::cout << "  Cropped point count: " << cloud_cropped->size() << std::endl;

  if (cloud_cropped->size() < 200)
  {
    PCL_ERROR("Cropped cloud has too few points, cannot segment plane\n");
    return -1;
  }

  // ===== 4. Iteratively segment planes, find target plane =====
  std::cout << "\nSearching for target plane (L=6130±200mm, W=770±200mm, RANSAC distance_threshold="
            << distance_threshold << ", iterations=" << ransac_iterations << ")..." << std::endl;

  PointCloudT::Ptr plane_cloud(new PointCloudT);
  pcl::ModelCoefficients coefficients;
  std::vector<LeftWall> left_walls;
  int seg_result = segmentTargetPlane(cloud_cropped, plane_cloud, coefficients,
                                       distance_threshold, ransac_iterations,
                                       left_walls,
                                       filter_outliers, outlier_radius, outlier_min_neighbors);
  if (seg_result < 0)
  {
    return -1;
  }
  bool exact_match = (seg_result == 1);

  // ===== 5. Compute plane coordinate system pose =====
  std::cout << "\nComputing plane coordinate system pose..." << std::endl;
  PlanePose pose;
  if (!computePlanePose(plane_cloud, coefficients, pose))
  {
    return -1;
  }
  printPose(pose);

  // ===== 6. Compute plane bounding box =====
  PlaneBBox bbox = computePlaneBBox(plane_cloud, pose);

  // ===== 7. Output plane equation =====
  printf("\n  === Plane Equation ===\n");
  printf("  %.6fx + %.6fy + %.6fz + %.6f = 0\n",
         coefficients.values[0], coefficients.values[1],
         coefficients.values[2], coefficients.values[3]);

  printf("\n  === Plane Bounding Box (in plane coordinate system) ===\n");
  printf("  x: [%.4f, %.4f]\n", bbox.x_min, bbox.x_max);
  printf("  y: [%.4f, %.4f]\n", bbox.y_min, bbox.y_max);
  printf("  z: [%.4f, %.4f]\n", bbox.z_min, bbox.z_max);

  // ===== 7.5. Left walls collected during plane segmentation =====
  std::cout << "\nTotal left walls (x < 0) detected: " << left_walls.size() << std::endl;

  if (!left_walls.empty())
  {
    printf("\n  === Left Wall Normals ===\n");
    Eigen::Vector3f avg_normal(0, 0, 0);
    for (const auto& wall : left_walls)
    {
      Eigen::Vector3f n(wall.coefficients.values[0],
                        wall.coefficients.values[1],
                        wall.coefficients.values[2]);
      n.normalize();
      printf("    Wall %d: [%+.6f, %+.6f, %+.6f]\n", wall.index, n.x(), n.y(), n.z());
      avg_normal += n;
    }
    avg_normal /= static_cast<float>(left_walls.size());
    printf("    Average: [%+.6f, %+.6f, %+.6f] (norm=%.6f)\n",
           avg_normal.x(), avg_normal.y(), avg_normal.z(), avg_normal.norm());
  

    // ===== 5.5 直接使用 Eigen 计算 baselink 在 map 下的位姿 =====
    Eigen::Vector3f T_bl_map = Eigen::Vector3f::Zero();
    Eigen::Matrix3f R_bl_map = Eigen::Matrix3f::Identity();
    
    {
      // 1) map -> net 的静态变换 (T_map_net)
      Eigen::Vector3f t_map_net(kNetLeftX, kNetLeftY, 0.0f);
      Eigen::Matrix3f R_map_net = Eigen::Matrix3f::Identity(); // 旋转为0

      // 2) baselink -> net 的变换 (T_baselink_net)
      Eigen::Vector3f t_bl_net = pose.position;
      // 使用左侧墙来修正网的角度
      Eigen::Matrix3f R_bl_net;
      Eigen::Vector3f x_axis(avg_normal.x(), avg_normal.y(), 0.0f); // 墙的法向量投影到XY平面
      x_axis.normalize();
      // Y 轴 = X 轴逆时针转90度
      Eigen::Vector3f y_axis(-x_axis.y(), x_axis.x(), 0.0f);
      y_axis.normalize();
      // Z 轴 = (0,0,1)
      Eigen::Vector3f z_axis(0.0f, 0.0f, 1.0f);
      R_bl_net.col(0) = x_axis;
      R_bl_net.col(1) = y_axis;
      R_bl_net.col(2) = z_axis;

      // 3) 计算 net -> baselink 的逆变换 (T_net_baselink)
      Eigen::Matrix3f R_net_bl = R_bl_net.transpose();
      Eigen::Vector3f t_net_bl = -R_net_bl * t_bl_net;

      // 4) 矩阵乘法：T_map_baselink = T_map_net * T_net_baselink
      R_bl_map = R_map_net * R_net_bl;
      T_bl_map = R_map_net * t_net_bl + t_map_net;

      float yaw = std::atan2(R_bl_map(1, 0), R_bl_map(0, 0));

      printf("\n  === baselink pose in map frame (Calculated via Eigen) ===\n");
      printf("    Translation (x, y, z): %.6f, %.6f, %.6f\n",
            T_bl_map.x(), T_bl_map.y(), T_bl_map.z());
      printf("    yaw: %.6f rad (%.3f deg)\n", yaw, yaw * 180.0f / M_PI);
      printf("    Rotation matrix (baselink -> map):\n");
      printf("      [%8.6f, %8.6f, %8.6f]\n",
            R_bl_map(0, 0), R_bl_map(0, 1), R_bl_map(0, 2));
      printf("      [%8.6f, %8.6f, %8.6f]\n",
            R_bl_map(1, 0), R_bl_map(1, 1), R_bl_map(1, 2));
      printf("      [%8.6f, %8.6f, %8.6f]\n",
            R_bl_map(2, 0), R_bl_map(2, 1), R_bl_map(2, 2));
    }

    // ===== 8. Visualization =====
    if (!no_viz)
    {
      visualize(cloud, cloud_cropped, plane_cloud, pose, bbox, axis_length, exact_match,
                T_bl_map, R_bl_map, left_walls);
    }

  }
  

  rclcpp::shutdown();
  return 0;
}
