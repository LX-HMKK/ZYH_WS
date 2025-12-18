import sys
import os
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
# from grasp_publisher.msg import GraspResult
from grasp_interfaces.msg import GraspResult
from std_msgs.msg import String
import numpy as np
import cv2
import time

sys.path.append("/home/zyh/ZYH_WS/src/graspnet-baseline-main")

# -------------- 从 robot.py 里导入必要函数 --------------
from kw.robot import (
    load_all_models,
    process_aligned_frames,
    generate_masks_auto,
    generate_masks,
    run_grasp_prediction_auto,
    run_grasp_prediction,
    Config
)


class GraspPublisher(Node):
    def __init__(self):
        super().__init__('grasp_publisher')
        self.pub = self.create_publisher(GraspResult, '/grasp_result', 10)
        self.subscription = self.create_subscription(
            String,
            'robot_status',
            self.status_callback,
            10)
        
        # 标志位，表示是否准备好进行下一次检测
        self.ready_for_next = True
        self.first_detection_done = False

        # -------------- 初始化模型 & 相机 --------------
        self.yolo_model, self.sam_predictor, self.grasp_net, self.device = load_all_models()
        self.pipeline, self.aligner, self.depth_scale = self.init_realsense()
        self.get_logger().info('GraspPublisher 启动，将自动进行抓取检测')
        
        # 初始化完成后立即执行第一次检测
        self.execute_detection()
        self.first_detection_done = True

    def status_callback(self, msg):
        """处理来自机器人的状态反馈"""
        if msg.data == "have backed":
            self.get_logger().info('收到机器人返回状态："have backed"，准备进行下一次检测')
            self.execute_detection()

    def init_realsense(self):
        import pyrealsense2 as rs
        # 添加重试机制和延时，让初始化更稳定
        max_retries = 5
        retry_delay = 1.0  # 延迟时间（秒）
        
        for attempt in range(max_retries):
            try:
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, *Config.CAMERA_RES, rs.format.rgb8, Config.CAMERA_FPS)
                cfg.enable_stream(rs.stream.depth, *Config.CAMERA_RES, rs.format.z16, Config.CAMERA_FPS)

                align_to = rs.stream.color
                aligner = rs.align(align_to)

                profile = pipeline.start(cfg)
                
                # 获取深度传感器和深度比例
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.get_logger().info(f'深度比例系数：{depth_scale:.6f} 米/像素')
                
                # 添加额外的等待时间，确保相机完全初始化
                time.sleep(3.0)
                
                self.get_logger().info(f'RealSense camera initialized successfully on attempt {attempt + 1}')
                return pipeline, aligner, depth_scale
                
            except Exception as e:
                self.get_logger().warn(f'Attempt {attempt + 1} failed to initialize RealSense camera: {str(e)}')
                if attempt < max_retries - 1:
                    self.get_logger().info(f'Waiting {retry_delay} seconds before retry...')
                    time.sleep(retry_delay)
                else:
                    self.get_logger().error('Failed to initialize RealSense camera after all retries')
                    raise e
    
    def cleanup_resources(self):
        """释放资源，参考robot.py中的实现方式"""
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
            self.get_logger().info('RealSense pipeline stopped and resources released')
        
        # 清理其他可能的资源
        import cv2
        cv2.destroyAllWindows()
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.get_logger().info('Resources cleanup completed')

    def execute_detection(self):
        """执行抓取检测"""
        # 检查是否已经执行过第一次检测
        if not self.first_detection_done or self.ready_for_next:
            self.ready_for_next = False  # 设置标志位防止重复执行
            self.get_logger().info("=== 开始抓取检测 ===")
        else:
            self.get_logger().warn("检测被跳过，因为系统尚未准备好进行下一次检测")
            return
        
        try:
            # 获取相机帧，增加超时时间到10秒
            frames = self.pipeline.wait_for_frames(timeout_ms=10000)
            
            # 处理对齐帧
            color_aligned, depth_aligned, depth_colormap = process_aligned_frames(
                frames, self.aligner, Config.USE_ROS_BAG
            )

            color_aligned=cv2.cvtColor(color_aligned, cv2.COLOR_RGB2BGR)
            
            # 保存对齐图像到临时路径
            color_path = '/tmp/aligned_color.png'
            depth_path = '/tmp/aligned_depth.png'
            
            # 保存彩色图（RGB）
            cv2.imwrite(color_path, cv2.cvtColor(color_aligned, cv2.COLOR_RGB2BGR))
            # 保存深度图（16位格式）
            cv2.imwrite(depth_path, depth_aligned.astype(np.uint16))
            self.get_logger().info(f"临时保存图像：{color_path}, {depth_path}")
            
            # 生成掩码
            try:
                sam_mask_path, yolo_mask_path, cls_name = generate_masks(
                    color_aligned, color_path, self.yolo_model, self.sam_predictor, self.device)
                if cls_name == 'None':
                    self.get_logger().warn('本次未检测到有效抓取，自动进入下一轮检测')
                    self.ready_for_next = True  # 恢复标志位
                    time.sleep(2.0)
                    self.execute_detection()  # 自动进入下一轮检测
                    return
            except UnboundLocalError as e:
                if "local variable 'cls_name' referenced before assignment" in str(e):
                    self.get_logger().warn('本次未检测到任何目标，自动进入下一轮检测')
                    self.ready_for_next = True  # 恢复标志位
                    time.sleep(2.0)
                    self.execute_detection()  # 自动进入下一轮检测
                    return
                else:
                    raise e
            # print(cls_name)
            # 选择掩码
            mask_path = yolo_mask_path if Config.MASK_CHOICE == 1 else sam_mask_path
            self.get_logger().info(f"使用掩码类型：{'YOLO扩展掩码' if Config.MASK_CHOICE == 1 else 'SAM分割掩码'}")
            
            # 执行抓取预测
            try:
                ret = run_grasp_prediction(self.grasp_net, color_path, depth_path, mask_path)
                # ret = run_grasp_prediction_auto(self.grasp_net, color_path, depth_path, mask_path)
                if ret is None:
                    self.get_logger().warn('本次未检测到有效抓取，等待下次检测')
                    self.ready_for_next = True  # 恢复标志位
                    time.sleep(2.0)
                    self.execute_detection()  # 自动进入下一轮检测
                    # return

                best_trans_cam, best_rot_mat_cam, best_width, best_pose_base, top_grasps = ret
                best_score = top_grasps[0].score

                # pos_base每位*1000
                pos_base = best_pose_base[:3] * 1000
                # euler_base每位*57.3
                euler_base = best_pose_base[3:] * 57.3

                # 填充消息
                msg = GraspResult()
                msg.trans_cam = best_trans_cam.tolist()
                msg.rot_cam_flat = best_rot_mat_cam.flatten().tolist()
                msg.width = float(best_width)
                msg.score = float(best_score)
                msg.pos_base = pos_base.tolist()     
                msg.euler_base = euler_base.tolist() 
                msg.cls_name = cls_name

                self.pub.publish(msg)
                self.get_logger().info(f'已发布 /grasp_result, 识别类名{cls_name}')
                self.get_logger().info(f'点位 {pos_base}, 欧拉角 {euler_base}')
                
            except ValueError as e:
                if "a must be greater than 0 unless no samples are taken" in str(e):
                    self.get_logger().warn('未检测到有效点云数据，可能是没有检测到物体')
                    self.ready_for_next = True  # 恢复标志位
                    time.sleep(2.0)
                    self.execute_detection()  # 自动进入下一轮检测
                else:
                    # 重新抛出其他ValueError异常
                    raise e
            except Exception as e:
                self.get_logger().error(f'处理过程中出现错误: {str(e)}')
                
        except Exception as e:
            self.get_logger().warn(f'获取相机帧时出错: {str(e)}')
            # 如果是超时错误，尝试重启相机
            if "timeout"or"Frame didn't arrive within" in str(e).lower():
                self.get_logger().warn("检测到超时错误，尝试重启相机...")
                self.restart_camera()
        finally:
            # 确保在任何情况下都恢复标志位
            self.ready_for_next = True
            
    def restart_camera(self):
        """重启相机连接"""
        try:
            self.get_logger().info("正在重启相机...")
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
                
            # 等待一段时间
            time.sleep(2.0)
            
            # 重新初始化相机
            self.pipeline, self.aligner, self.depth_scale = self.init_realsense()
            self.get_logger().info("相机重启成功")
        except Exception as e:
            self.get_logger().error(f"相机重启失败: {str(e)}")

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