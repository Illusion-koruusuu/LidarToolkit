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
        Node(
            package='LidarToolkit',
            executable='cluster_rviz',
            name='cluster_rviz',
            output='screen',
            parameters=[{
                'pcd_path': pcd_path,
                'frame_id': frame_id,
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
