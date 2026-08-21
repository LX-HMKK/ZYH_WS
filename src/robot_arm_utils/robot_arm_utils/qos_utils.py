#coding=utf-8
"""通用 ROS 2 QoS 配置。"""
from rclpy.qos import QoSProfile, ReliabilityPolicy


def reliable_qos(depth: int = 10) -> QoSProfile:
    """返回可靠的 QoSProfile。"""
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=depth)
