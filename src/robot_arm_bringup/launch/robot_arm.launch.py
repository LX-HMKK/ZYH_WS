from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """启动机械臂抓取系统的三个核心节点。

    各节点通过 ROBOARM_WS 环境变量解析仓库根目录下的工具和配置文件。
    使用示例：
        export ROBOARM_WS=/home/zyh/ZYH_WS
        ros2 launch robot_arm_bringup robot_arm.launch.py
    """
    return LaunchDescription([
        Node(
            package="grasp_publisher",
            executable="grasp_node",
            name="grasp_publisher",
            output="screen",
        ),
        Node(
            package="codroid_node",
            executable="codroid_io",
            name="codroid_io",
            output="screen",
        ),
        Node(
            package="codroid_node",
            executable="codroid_move_test",
            name="codroid_move_test",
            output="screen",
        ),
    ])
