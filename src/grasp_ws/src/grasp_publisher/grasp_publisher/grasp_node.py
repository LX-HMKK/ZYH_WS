#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grasp_node  带“相机重启后自动继续检测”功能
"""
import sys
import os
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
import pyrealsense2 as rs
from grasp_interfaces.msg import GraspResult
from std_msgs.msg import String
import numpy as np
import cv2
import time

sys.path.append("/home/zyh/ZYH_WS/src/graspnet-baseline-main")
from kw.robot import (
    load_all_models,
    process_aligned_frames,
    generate_masks_auto,
    run_grasp_prediction_auto,
    Config
)


class GraspPublisher(Node):
    def __init__(self):
        super().__init__('grasp_publisher')
        self.pub = self.create_publisher(GraspResult, '/grasp_result', 10)
        self.create_subscription(String, 'robot_status', self.status_callback, 10)

        # 控制流程的标志位
        self.ready_for_next = True
        self.first_detection_done = False

        # 相机重启次数控制
        self.restart_cnt = 0
        self.MAX_RESTART = 3

        # 模型与相机初始化
        self.yolo_model, self.sam_predictor, self.grasp_net, self.device = load_all_models()
        self.pipeline, self.aligner, self.depth_scale = self.init_realsense()
        self.get_logger().info('GraspPublisher 启动，将自动进行抓取检测')

        # 启动后先跑一帧
        self.execute_detection()
        self.first_detection_done = True

    # ---------------- 相机初始化 ----------------
    def init_realsense(self):
        max_retries = 5
        retry_delay = 1.0
        for attempt in range(max_retries):
            try:
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, *Config.CAMERA_RES, rs.format.rgb8, 15)
                cfg.enable_stream(rs.stream.depth, *Config.CAMERA_RES, rs.format.z16, 15)
                aligner = rs.align(rs.stream.color)
                profile = pipeline.start(cfg)
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                self.get_logger().info(f'深度比例系数：{depth_scale:.6f} 米/像素')
                time.sleep(4.0)          # 等固件彻底 ready
                self.get_logger().info(f'RealSense camera initialized successfully on attempt {attempt + 1}')
                return pipeline, aligner, depth_scale
            except Exception as e:
                self.get_logger().warn(f'Attempt {attempt + 1} failed to initialize RealSense camera: {str(e)}')
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self.get_logger().error('Failed to initialize RealSense camera after all retries')
                    raise e

    # ---------------- 机器人状态回调 ----------------
    def status_callback(self, msg):
        if msg.data == "have backed":
            self.get_logger().info('收到机器人返回状态："have backed"，准备进行下一次检测')
            self.execute_detection()

    # ---------------- 资源释放 ----------------
    def cleanup_resources(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
            self.get_logger().info('RealSense pipeline stopped')
        cv2.destroyAllWindows()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.get_logger().info('Resources cleanup completed')

    # ---------------- 重启相机 ----------------
    def restart_camera(self):
        try:
            self.get_logger().info("正在重启相机...")
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
                time.sleep(1.0)
                del self.pipeline
            time.sleep(2.0)                 # 等硬件掉线
            self.pipeline, self.aligner, self.depth_scale = self.init_realsense()
            time.sleep(2.0)
            self.get_logger().info("相机重启成功")
        except Exception as e:
            self.get_logger().error(f"相机重启失败: {str(e)}")

    # ---------------- 抓取检测主流程 ----------------
    def execute_detection(self):
        # 新一帧开始，重启计数器清零
        self.restart_cnt = 0
        if not self.first_detection_done or self.ready_for_next:
            self.ready_for_next = False
            self.get_logger().info("=== 开始抓取检测 ===")
        else:
            self.get_logger().warn("检测被跳过，系统尚未准备好")
            return

        try:
            time.sleep(0.5)
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            color_aligned, depth_aligned, _ = process_aligned_frames(
                frames, self.aligner, Config.USE_ROS_BAG)
            color_aligned = cv2.cvtColor(color_aligned, cv2.COLOR_RGB2BGR)

            color_path = '/tmp/aligned_color.png'
            depth_path = '/tmp/aligned_depth.png'
            cv2.imwrite(color_path, color_aligned)
            cv2.imwrite(depth_path, depth_aligned.astype(np.uint16))

            # 生成掩码
            try:
                sam_mask_path, yolo_mask_path, cls_name = generate_masks_auto(
                    color_aligned, color_path, self.yolo_model, self.sam_predictor, self.device)
                if cls_name == 'None':
                    self.get_logger().warn('本次未检测到有效抓取，自动进入下一轮检测')
                    self.ready_for_next = True
                    time.sleep(2.0)
                    self.execute_detection()
                    return
            except UnboundLocalError:
                self.get_logger().warn('本次未检测到任何目标，自动进入下一轮检测')
                self.ready_for_next = True
                time.sleep(2.0)
                self.execute_detection()
                return

            mask_path = yolo_mask_path if Config.MASK_CHOICE == 1 else sam_mask_path
            self.get_logger().info(f"使用掩码类型：{'YOLO扩展掩码' if Config.MASK_CHOICE == 1 else 'SAM分割掩码'}")

            # 抓取预测
            ret = run_grasp_prediction_auto(self.grasp_net, color_path, depth_path, mask_path)
            if ret is None:
                self.get_logger().warn('本次未检测到有效抓取，等待下次检测')
                self.ready_for_next = True
                time.sleep(2.0)
                self.execute_detection()
                return

            best_trans_cam, best_rot_mat_cam, best_width, best_pose_base, top_grasps = ret
            best_score = top_grasps[0].score
            pos_base = best_pose_base[:3] * 1000
            euler_base = best_pose_base[3:] * 57.3

            msg = GraspResult()
            msg.trans_cam = best_trans_cam.tolist()
            msg.rot_cam_flat = best_rot_mat_cam.flatten().tolist()
            msg.width = float(best_width)
            msg.score = float(best_score)
            msg.pos_base = pos_base.tolist()
            msg.euler_base = euler_base.tolist()
            msg.cls_name = cls_name
            self.pub.publish(msg)
            self.get_logger().info(f'已发布 /grasp_result, 识别类名 {cls_name}')
            self.get_logger().info(f'点位 {pos_base}, 欧拉角 {euler_base}')

        except Exception as e:
            self.get_logger().warn(f'获取相机帧时出错: {str(e)}')
            # 关键：重启后继续本帧，而非退出
            if self.restart_cnt < self.MAX_RESTART and \
               ('timeout' in str(e).lower() or "frame didn't arrive" in str(e).lower()):
                self.restart_cnt += 1
                self.get_logger().warn(f'第 {self.restart_cnt} 次重启相机并重新检测...')
                self.restart_camera()
                self.execute_detection()   # 递归继续
                return
            else:
                self.get_logger().error('连续重启仍失败，放弃本帧')
        finally:
            self.ready_for_next = True


# ---------------- main ----------------
def main(args=None):
    rclpy.init(args=args)
    node = GraspPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n=== 程序退出 ===')
    finally:
        node.cleanup_resources()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()