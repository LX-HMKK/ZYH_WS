from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """启动机械臂抓取系统全部四个节点（含语音/LLM 节点）。"""
    return LaunchDescription([
        Node(
            package="vision_grasp",
            executable="grasp_node",
            name="vision_grasp",
            output="screen",
        ),
        Node(
            package="arm_control",
            executable="io_node",
            name="arm_io",
            output="screen",
        ),
        Node(
            package="arm_control",
            executable="motion_node",
            name="motion_node",
            output="screen",
        ),
        Node(
            package="llm_voice",
            executable="llm_node",
            name="llm_node",
            output="screen",
        ),
    ])
