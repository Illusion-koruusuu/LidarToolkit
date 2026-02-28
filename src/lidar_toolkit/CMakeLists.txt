cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(LidarToolkit)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL 1.5 REQUIRED)
set(CMAKE_CXX_STANDARD 17)

include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(EXECUTABLES
	icp
	merge
	tailor
	cluster_extraction
	cluster_rviz
)

foreach(exec ${EXECUTABLES})
	add_executable(${exec} ${exec}.cpp)
	target_link_libraries(${exec} ${PCL_LIBRARIES})
	ament_target_dependencies(${exec}
		rclcpp
		sensor_msgs
		visualization_msgs
		pcl_conversions
	)
endforeach()

# Install targets so colcon can install them into the install space
install(TARGETS ${EXECUTABLES}
	RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
	launch
	rviz
	DESTINATION share/${PROJECT_NAME}
)

ament_package()