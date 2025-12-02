cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(icp_test)

find_package(PCL 1.5 REQUIRED)
set(CMAKE_CXX_STANDARD 17)

add_executable (icp_test icp.cpp)
add_executable (merge merge.cpp)
add_executable (tailor tailor.cpp)

target_link_libraries (icp_test ${PCL_LIBRARIES})
target_link_libraries (merge ${PCL_LIBRARIES})
target_link_libraries (tailor ${PCL_LIBRARIES})