import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 创建保存目录
output_dir = "captured_images"
color_dir = os.path.join(output_dir, "color_images")
depth_dir = os.path.join(output_dir, "depth_images")
os.makedirs(color_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# 初始化相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

image_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 转换图像数据
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 显示图像
        cv2.imshow('Color', color_image)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        cv2.imshow('Depth', depth_colormap)

        # 按键处理
        key = cv2.waitKey(1)
        if key == ord('s'):
            # 保存彩色图像
            color_path = os.path.join(color_dir, f"color_{image_count:04d}.png")
            cv2.imwrite(color_path, color_image)
            
            # 保存深度图像（16位PNG）
            depth_path = os.path.join(depth_dir, f"depth_{image_count:04d}.png")
            cv2.imwrite(depth_path, depth_image.astype(np.uint16))  # 确保16位格式
            
            print(f"已保存：{color_path} 和 {depth_path}")  # 修复print语句
            image_count += 1  # 计数器递增
            
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()