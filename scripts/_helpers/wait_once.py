#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等待 /robot_status 发布 'have backed'，或超时退出。

用法：
    python3 wait_once.py [TIMEOUT_SECONDS]

退出码：
    0 - 收到 'have backed'
    1 - 超时
"""
import sys
import time

import rclpy
from std_msgs.msg import String


def main():
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0

    rclpy.init()
    node = rclpy.create_node("grasp_once_monitor")

    done = False

    def cb(msg: String):
        nonlocal done
        if msg.data.strip().lower() == "have backed":
            node.get_logger().info("收到状态：have backed，抓取周期完成")
            done = True

    node.create_subscription(String, "/robot_status", cb, 10)

    start = time.time()
    while rclpy.ok() and not done and (time.time() - start) < timeout:
        rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()

    if done:
        print("[OK] 一次抓取完成")
        sys.exit(0)
    else:
        print(f"[TIMEOUT] {timeout}s 内未收到 have backed")
        sys.exit(1)


if __name__ == "__main__":
    main()
