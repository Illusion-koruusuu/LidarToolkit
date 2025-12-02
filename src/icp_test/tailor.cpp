/*
 * tailor.cpp
 * 用法: ./build/icp_test/tailor <input_path>
 * <input_path> 可以是单个 PCD 文件,也可以是包含多个 PCD 文件的目录.
 * 程序会处理所有输入 PCD 文件,去除 Z 轴小于 2.0 米的点,并将结果保存到 ./output 目录下.
 */

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace fs = std::filesystem;

int process_pcd(const fs::path &inpath) {
    if (!fs::exists(inpath)) {
        std::cerr << "Input path does not exist: " << inpath << '\n';
        return 1;
    }
    if (inpath.extension() != ".pcd") return 0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(inpath.string(), *cloud) == -1) {
        std::cerr << "Failed to read PCD or unsupported point type: " << inpath << '\n';
        return 1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    out->points.reserve(cloud->points.size());
    for (const auto &p : cloud->points) {
        if (p.z >= 2.0f) out->points.push_back(p);
    }
    out->width = static_cast<uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = false;

    fs::create_directories("output");
    fs::path outpath = fs::path("output") / inpath.filename();

    if (pcl::io::savePCDFileBinary(outpath.string(), *out) != 0) {
        std::cerr << "Failed to write output PCD: " << outpath << '\n';
        return 1;
    }

    std::cout << "Wrote: " << outpath << " (" << out->points.size() << " points)\n";
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_path>\n";
        std::cerr << "  <input_path> can be a .pcd file or a directory containing .pcd files.\n";
        return 1;
    }

    fs::path p(argv[1]);
    if (!fs::exists(p)) {
        std::cerr << "Path not found: " << p << '\n';
        return 1;
    }

    int rc = 0;
    if (fs::is_directory(p)) {
        for (const auto &entry : fs::directory_iterator(p)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() == ".pcd") {
                std::cout << "Processing: " << entry.path() << '\n';
                rc |= process_pcd(entry.path());
            }
        }
    } else {
        if (p.extension() != ".pcd") {
            std::cerr << "Only .pcd files are supported.\n";
            return 1;
        }
        rc = process_pcd(p);
    }

    return rc;
}
