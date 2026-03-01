from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('LidarToolkit')
    rviz_config = os.path.join(pkg_share, 'rviz', 'cluster_rviz.rviz')

    pcd_path = LaunchConfiguration('pcd_path')
    frame_id = LaunchConfiguration('frame_id')

    publish_rate_hz = LaunchConfiguration('publish_rate_hz')
    voxel_leaf_size = LaunchConfiguration('voxel_leaf_size')
    plane_distance_threshold = LaunchConfiguration('plane_distance_threshold')
    plane_max_iterations = LaunchConfiguration('plane_max_iterations')
    cluster_tolerance = LaunchConfiguration('cluster_tolerance')
    min_cluster_size = LaunchConfiguration('min_cluster_size')
    max_cluster_size = LaunchConfiguration('max_cluster_size')
    publish_filtered_cloud = LaunchConfiguration('publish_filtered_cloud')

    return LaunchDescription([
        DeclareLaunchArgument(
            'pcd_path',
            default_value='table_scene_lms400.pcd',
            description='Input PCD file path',
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='map',
            description='Frame id for published messages',
        ),
        DeclareLaunchArgument(
            'publish_rate_hz',
            default_value='1.0',
            description='Publish rate (Hz)',
        ),
        DeclareLaunchArgument(
            'voxel_leaf_size',
            default_value='0.05',
            description='VoxelGrid leaf size (m)',
        ),
        DeclareLaunchArgument(
            'plane_distance_threshold',
            default_value='0.02',
            description='RANSAC plane distance threshold (m)',
        ),
        DeclareLaunchArgument(
            'plane_max_iterations',
            default_value='100',
            description='RANSAC max iterations',
        ),
        DeclareLaunchArgument(
            'cluster_tolerance',
            default_value='0.1',
            description='Euclidean cluster tolerance (m)',
        ),
        DeclareLaunchArgument(
            'min_cluster_size',
            default_value='100',
            description='Min points per cluster',
        ),
        DeclareLaunchArgument(
            'max_cluster_size',
            default_value='25000',
            description='Max points per cluster',
        ),
        DeclareLaunchArgument(
            'publish_filtered_cloud',
            default_value='true',
            description='Whether to publish /filtered_cloud (true/false)',
        ),
        Node(
            package='LidarToolkit',
            executable='cluster_rviz',
            name='cluster_rviz',
            output='screen',
            parameters=[{
                'pcd_path': pcd_path,
                'frame_id': frame_id,
                'publish_rate_hz': publish_rate_hz,
                'voxel_leaf_size': voxel_leaf_size,
                'plane_distance_threshold': plane_distance_threshold,
                'plane_max_iterations': plane_max_iterations,
                'cluster_tolerance': cluster_tolerance,
                'min_cluster_size': min_cluster_size,
                'max_cluster_size': max_cluster_size,
                'publish_filtered_cloud': publish_filtered_cloud,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])
