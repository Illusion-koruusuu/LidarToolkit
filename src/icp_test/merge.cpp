/*
 * merge.cpp
 * 合并 ./merge 目录下的所有 PCD/PLY 点云为一个 PCD 文件,输出到./output 目录下.
 * 用法: ./build/icp_test/merge
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <ctime>
#include <iomanip>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

int main(int argc, char** argv)
{
  std::string out_dir = "./output";
  std::string input_dir = "./merge";

  namespace fs = std::filesystem;
  fs::path dir(input_dir);
  if (!fs::exists(dir) || !fs::is_directory(dir))
  {
    std::cerr << "Input directory does not exist: " << input_dir << std::endl;
    return -1;
  }

  PointCloudT::Ptr merged(new PointCloudT);
  size_t files_processed = 0;

  for (const auto &entry : fs::directory_iterator(dir))
  {
    if (!entry.is_regular_file()) continue;
    fs::path p = entry.path();
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    PointCloudT::Ptr tmp(new PointCloudT);
    int ret = -1;
    if (ext == ".pcd")
    {
      ret = pcl::io::loadPCDFile(p.string(), *tmp);
    }
    else if (ext == ".ply")
    {
      ret = pcl::io::loadPLYFile(p.string(), *tmp);
    }
    else
    {
      continue; // skip unknown extensions
    }

    if (ret < 0)
    {
      std::cerr << "Failed to load " << p.string() << std::endl;
      continue;
    }

    std::cout << "Loaded " << p.filename().string() << " (" << tmp->size() << " points)" << std::endl;
    *merged += *tmp;
    ++files_processed;
  }

  if (files_processed == 0)
  {
    std::cerr << "No PCD/PLY files found in " << input_dir << std::endl;
    return -1;
  }

  // build output filename
  fs::path out_dir_(out_dir);
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::ostringstream ss;
  ss << out_dir_.string();
  if (!out_dir_.string().empty() && out_dir_.string().back() != '/') ss << '/';
  ss << "merged_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".pcd";
  std::string out_path = ss.str();

  if (pcl::io::savePCDFileBinary(out_path, *merged) < 0)
  {
    std::cerr << "Failed to save merged cloud to " << out_path << std::endl;
    return -1;
  }

  std::cout << "Saved merged cloud to " << out_path << " (" << merged->size() << " points, " << files_processed << " files)" << std::endl;
  return 0;
}
