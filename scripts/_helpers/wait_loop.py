#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""监控 /grasp_result：完成至少一次抓取后，若长时间无新目标则退出。

用法：
    python3 wait_loop.py [IDLE_TIMEOUT_SECONDS] [MAX_TOTAL_SECONDS]

退出码：
    0 - 达到空闲阈值（无目标）或达到最大运行时间
"""
import sys
import time

import rclpy
from arm_interfaces.msg import GraspResult


def main():
    idle_timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    max_total = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0

    rclpy.init()
    node = rclpy.create_node("grasp_loop_monitor")

    first_grasp_time = None
    last_grasp_time = None

    def cb(msg: GraspResult):
        nonlocal first_grasp_time, last_grasp_time
        now = time.time()
        last_grasp_time = now
        if first_grasp_time is None:
            first_grasp_time = now
            node.get_logger().info(
                f"检测到第一个目标：{msg.cls_name}，后续 {idle_timeout}s 无新目标将自动停止"
            )
        else:
            node.get_logger().info(f"检测到新目标：{msg.cls_name}")

    node.create_subscription(GraspResult, "/grasp_result", cb, 10)

    start = time.time()
    while rclpy.ok() and (time.time() - start) < max_total:
        rclpy.spin_once(node, timeout_sec=1.0)
        now = time.time()

        # 完成至少一次抓取后，若超过空闲阈值无新目标则退出
        if first_grasp_time is not None and (now - last_grasp_time) > idle_timeout:
            node.get_logger().info(f"{idle_timeout}s 未检测到新目标，自动停止")
            break

    node.destroy_node()
    rclpy.shutdown()
    print("[OK] 多次抓取结束")


if __name__ == "__main__":
    main()
