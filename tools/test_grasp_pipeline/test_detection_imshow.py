#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单帧抓取检测测试（交互式可视化）。

加载 YOLO + SAM + GraspNet，从 RealSense 采集一帧，
显示 SAM 分割结果与 GraspNet 抓取位姿（Open3D）。
运行前请确保已配置 ROBOARM_WS 并放置模型权重到 assets/。
"""
import os
import sys
import tempfile
import time

import numpy as np
import cv2
import pyrealsense2 as rs

sys.path.append(f"{os.environ.get('ROBOARM_WS', '/home/zyh/ZYH_WS')}/src")
from arm_utils import get_workspace_root
from grasp_pipeline.config import Config
from grasp_pipeline.core.model_manager import ModelManager
from grasp_pipeline.core.frame_processor import FrameProcessor
from grasp_pipeline.core.object_segmentor import ObjectSegmentor
from grasp_pipeline.core.grasp_predictor import GraspPredictor
from grasp_pipeline.core.camera_driver import RealSenseCamera


def main():
    print("=== 抓取检测交互式测试 ===")
    print("按 's' 保存当前帧并进入检测；按 'q' 或 ESC 退出相机预览")

    # 加载模型
    model_manager = ModelManager()
    yolo_model, sam_predictor, grasp_net, device = model_manager.load_all()

    frame_processor = FrameProcessor()
    segmentor = ObjectSegmentor(yolo_model, sam_predictor, device)
    predictor = GraspPredictor(grasp_net)

    camera = RealSenseCamera(config=Config, color_format=rs.format.rgb8)
    camera.start()

    try:
        while True:
            frames = camera.get_aligned_frames(timeout_ms=2000)
            color_aligned, depth_aligned, depth_colormap = frame_processor.process_aligned_frames(
                frames, camera.aligner, Config.USE_ROS_BAG
            )

            # 预览：左侧彩色，右侧深度伪彩
            display = np.hstack((color_aligned, depth_colormap))
            cv2.imshow("RealSense Preview (RGB | Depth)", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("用户退出")
                return
            if key == ord("s"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    # 保存临时图像（与 grasp_node 保持一致：存为 BGR）
    color_bgr = cv2.cvtColor(color_aligned, cv2.COLOR_RGB2BGR)
    with tempfile.NamedTemporaryFile(suffix="_color.png", delete=False) as color_f, \
         tempfile.NamedTemporaryFile(suffix="_depth.png", delete=False) as depth_f:
        color_path = color_f.name
        depth_path = depth_f.name

    cv2.imwrite(color_path, color_bgr)
    cv2.imwrite(depth_path, depth_aligned.astype(np.uint16))
    print(f"临时图像已保存：\n  彩色：{color_path}\n  深度：{depth_path}")

    # 交互式分割（显示 SAM 结果窗口）
    try:
        sam_mask_path, yolo_mask_path, cls_name = segmentor.generate_masks(
            color_aligned, color_path, interactive=True
        )
    except Exception as e:
        print(f"分割失败：{e}")
        return

    if cls_name == "None" or cls_name == "unknown":
        print("未检测到有效目标，测试结束")
        return

    mask_path = yolo_mask_path if Config.MASK_CHOICE == 1 else sam_mask_path
    print(f"使用掩码：{mask_path}")

    # 交互式抓取预测（显示 Open3D 抓取结果）
    try:
        ret = predictor.predict(color_path, depth_path, mask_path, visualize=True)
        if ret is None:
            print("GraspNet 未预测到有效抓取")
        else:
            best_trans_cam, best_rot_mat_cam, best_width, best_pose_base, top_grasps = ret
            print("\n=== 测试结果 ===")
            print(f"识别类别：{cls_name}")
            print(f"基坐标系位姿 [x,y,z,rx,ry,rz]：{best_pose_base}")
            print(f"抓取宽度：{best_width:.4f} m")
    except Exception as e:
        print(f"抓取预测失败：{e}")

    # 清理
    try:
        os.unlink(color_path)
        os.unlink(depth_path)
    except OSError:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cv2.destroyAllWindows()
