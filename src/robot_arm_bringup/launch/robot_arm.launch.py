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
            package="vision_grasp",
            executable="grasp_node",
            name="vision_grasp",
            output="screen",
        ),
        Node(
            package="robot_arm_control",
            executable="io_node",
            name="robot_arm_io",
            output="screen",
        ),
        Node(
            package="robot_arm_control",
            executable="motion_node",
            name="motion_node",
            output="screen",
        ),
    ])
