#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grasp_node  带“相机重启后自动继续检测”功能
"""
import os
import sys
import tempfile
from enum import Enum, auto
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from arm_interfaces.msg import GraspResult
from std_msgs.msg import String
import numpy as np
import cv2
import time


sys.path.append(f"{os.environ.get('ROBOARM_WS', '/home/zyh/ZYH_WS')}/src")
from arm_utils import get_workspace_root
from grasp_pipeline import Config, ModelManager, ObjectSegmentor, GraspPredictor
from vision_grasp.camera_driver import RealSenseCamera
from vision_grasp.frame_processor import FrameProcessor


class _DetectionStatus(Enum):
    SUCCESS = auto()
    NO_TARGET = auto()
    NO_GRASP = auto()
    CAMERA_RETRY = auto()
    FATAL = auto()


class GraspPublisher(Node):
    def __init__(self):
        super().__init__('vision_grasp')
        self.pub = self.create_publisher(GraspResult, '/grasp_result', 10)
        self.create_subscription(String, 'robot_status', self.status_callback, 10)

        # 声明 ROS 参数，允许覆盖配置文件路径
        self.declare_parameter('grasp_config_path', '')
        config_path = self.get_parameter('grasp_config_path').value
        config_path = config_path if config_path else None
        self.config = Config.load(config_path)

        # 控制流程的标志位
        self.ready_for_next = True
        self.first_detection_done = False

        # 相机重启次数控制
        self.restart_cnt = 0
        self.MAX_RESTART = 3

        # 模型与相机初始化
        self.model_manager = ModelManager(self.config)
        self.yolo_model, self.sam_predictor, self.grasp_net, self.device = self.model_manager.load_all()
        self.frame_processor = FrameProcessor()
        self.segmentor = ObjectSegmentor(self.yolo_model, self.sam_predictor, self.device, self.config)
        self.predictor = GraspPredictor(self.grasp_net, config=self.config, yolo_model=self.yolo_model)
        self.camera = RealSenseCamera(config=self.config, logger=self.get_logger())
        self.camera.start()
        self.get_logger().info('GraspPublisher 启动，将自动进行抓取检测')

        # 启动后先跑一帧
        self.execute_detection()
        self.first_detection_done = True

    # ---------------- 机器人状态回调 ----------------
    def status_callback(self, msg):
        if msg.data == "have backed":
            self.get_logger().info('收到机器人返回状态："have backed"，准备进行下一次检测')
            self.execute_detection()

    # ---------------- 资源释放 ----------------
    def cleanup_resources(self):
        self.camera.stop()
        cv2.destroyAllWindows()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.get_logger().info('Resources cleanup completed')

    # ---------------- 重启相机 ----------------
    def restart_camera(self):
        try:
            self.camera.restart()
        except Exception as e:
            self.get_logger().error(f"相机重启失败: {str(e)}")

    # ---------------- 抓取检测主流程 ----------------
    def _run_once(self):
        """执行一次抓取检测，返回状态与结果。"""
        try:
            time.sleep(0.5)
            frames = self.camera.get_aligned_frames(timeout_ms=2000)
            color_aligned, depth_aligned, _ = self.frame_processor.process_aligned_frames(
                frames, self.camera.aligner, self.config.USE_ROS_BAG)
            color_aligned = cv2.cvtColor(color_aligned, cv2.COLOR_RGB2BGR)

            with tempfile.NamedTemporaryFile(suffix="_color.png", delete=False) as color_f, \
                 tempfile.NamedTemporaryFile(suffix="_depth.png", delete=False) as depth_f:
                color_path = color_f.name
                depth_path = depth_f.name
            cv2.imwrite(color_path, color_aligned)
            cv2.imwrite(depth_path, depth_aligned.astype(np.uint16))

            # 生成掩码
            try:
                sam_mask_path, yolo_mask_path, cls_name = self.segmentor.generate_masks_auto(
                    color_aligned, color_path)
                if cls_name == 'None':
                    self.get_logger().warn('本次未检测到有效抓取，自动进入下一轮检测')
                    return _DetectionStatus.NO_TARGET, None
            except UnboundLocalError:
                self.get_logger().warn('本次未检测到任何目标，自动进入下一轮检测')
                return _DetectionStatus.NO_TARGET, None

            mask_path = yolo_mask_path if self.config.MASK_CHOICE == 1 else sam_mask_path
            self.get_logger().info(f"使用掩码类型：{'YOLO扩展掩码' if self.config.MASK_CHOICE == 1 else 'SAM分割掩码'}")

            # 抓取预测
            ret = self.predictor.predict_auto(color_path, depth_path, mask_path)
            if ret is None:
                self.get_logger().warn('本次未检测到有效抓取，等待下次检测')
                return _DetectionStatus.NO_GRASP, None

            best_trans_cam, best_rot_mat_cam, best_width, best_pose_base, top_grasps = ret
            best_score = top_grasps[0].score
            pos_base = best_pose_base[:3] * 1000
            euler_base = best_pose_base[3:] * 57.3

            msg = GraspResult()
            msg.stamp = self.get_clock().now().to_msg()
            msg.trans_cam = best_trans_cam.tolist()
            msg.rot_cam_flat = best_rot_mat_cam.flatten().tolist()
            msg.width = float(best_width)
            msg.score = float(best_score)
            msg.pos_base = pos_base.tolist()
            msg.euler_base = euler_base.tolist()
            msg.cls_name = cls_name
            return _DetectionStatus.SUCCESS, msg

        except Exception as e:
            self.get_logger().warn(f'获取相机帧时出错: {str(e)}')
            if 'timeout' in str(e).lower() or "frame didn't arrive" in str(e).lower():
                return _DetectionStatus.CAMERA_RETRY, None
            return _DetectionStatus.FATAL, None

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
            while True:
                status, result = self._run_once()
                if status == _DetectionStatus.SUCCESS:
                    self.pub.publish(result)
                    self.get_logger().info(f'已发布 /grasp_result, 识别类名 {result.cls_name}')
                    self.get_logger().info(f'点位 {result.pos_base}, 欧拉角 {result.euler_base}')
                    break
                elif status in (_DetectionStatus.NO_TARGET, _DetectionStatus.NO_GRASP):
                    time.sleep(2.0)
                    continue
                elif status == _DetectionStatus.CAMERA_RETRY:
                    if self.restart_cnt < self.MAX_RESTART:
                        self.restart_cnt += 1
                        self.get_logger().warn(f'第 {self.restart_cnt} 次重启相机并重新检测...')
                        self.restart_camera()
                        continue
                    else:
                        self.get_logger().error('连续重启仍失败，放弃本帧')
                        break
                elif status == _DetectionStatus.FATAL:
                    self.get_logger().error('检测失败，放弃本帧')
                    break
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
