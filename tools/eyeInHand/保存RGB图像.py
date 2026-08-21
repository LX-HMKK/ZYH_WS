#coding=utf-8
import sys
import os
import numpy as np
import cv2


def get_workspace_root() -> str:
    """返回仓库根目录，优先读取 ROBOARM_WS 环境变量。"""
    return os.environ.get("ROBOARM_WS", "/home/zyh/ZYH_WS")


WORKSPACE_ROOT = get_workspace_root()
sys.path.append(f"{WORKSPACE_ROOT}/src")

import pyrealsense2 as rs
from grasp_pipeline.core.camera_driver import RealSenseCamera


# 配置参数
output_dir = f"{WORKSPACE_ROOT}/eyeInHand/images"
os.makedirs(output_dir, exist_ok=True)
USE_ROS_BAG = False
ALIGN_TO_COLOR = True  # 对齐到彩色图像流


def main():
    save_count = 0

    # 使用 RealSenseCamera 封装，彩色流格式为 BGR8
    camera = RealSenseCamera(
        color_format=rs.format.bgr8,
        startup_delay=1.0,
    )
    camera.start()

    try:
        while True:
            # 获取对齐后的帧
            aligned_frames = camera.get_aligned_frames(timeout_ms=1000)

            # 获取彩色帧（BGR格式）
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # 获取深度帧并着色
            depth_frame = aligned_frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.05),
                cv2.COLORMAP_JET,
            )

            # 显示图像（左侧BGR彩色，右侧深度）
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('Aligned Frames', images)

            # 键盘控制
            key = cv2.waitKey(1)
            if key == ord('s'):  # 保存当前帧
                save_path = os.path.join(output_dir, f"{save_count:02d}.jpg")
                cv2.imwrite(save_path, color_image)
                print(f"Saved: {save_path}")
                save_count += 1
            elif key in [27, ord('q')]:  # ESC或q退出
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
