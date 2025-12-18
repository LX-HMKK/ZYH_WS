#coding=utf-8
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# 配置参数
output_dir = "/home/zyh/ZYH_WS/eyeInHand/images"
os.makedirs(output_dir, exist_ok=True)
USE_ROS_BAG = False
ALIGN_TO_COLOR = True  # 对齐到彩色图像流

def get_aligned_frames(pipeline, align):
    """获取对齐后的帧"""
    frames = pipeline.wait_for_frames(1000)
    aligned_frames = align.process(frames)
    return aligned_frames

def main():
    save_count = 0
    # 配置管道
    pipeline = rs.pipeline()
    config = rs.config()

    if USE_ROS_BAG:
        config.enable_device_from_file("666.bag")
    else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    
    # 设置对齐方式
    align = rs.align(rs.stream.color if ALIGN_TO_COLOR else rs.stream.depth)
    
    # 启动管道
    profile = pipeline.start(config)

    try:
        while True:
            # 获取对齐后的帧
            aligned_frames = get_aligned_frames(pipeline, align)
            
            # 获取彩色帧（确保是BGR格式）
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())  # 已经是BGR格式
            
            # 获取深度帧并着色
            depth_frame = aligned_frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.05), 
                cv2.COLORMAP_JET
            )
            
            # 显示图像（左侧BGR彩色，右侧深度）
            images = np.hstack((color_image, depth_colormap))
            cv2.namedWindow('Aligned Frames', cv2.WINDOW_AUTOSIZE)
            cv2.resizeWindow('Aligned Frames', 1280, 720)
            cv2.imshow('Aligned Frames', color_image)
            
            # 键盘控制
            key = cv2.waitKey(1)
            if key == ord('s'):  # 保存当前帧
                # 使用带微秒的时间戳作为文件名，若发生碰撞则添加序号后缀
                def timestamped_filename(directory, ext='.jpg'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{ts}{ext}"
                    path = os.path.join(directory, filename)
                    idx = 1
                    while os.path.exists(path):
                        filename = f"{ts}_{idx}{ext}"
                        path = os.path.join(directory, filename)
                        idx += 1
                    return path

                save_path = timestamped_filename(output_dir, '.jpg')
                cv2.imwrite(save_path, color_image)  # 直接保存BGR格式
                print(f"Saved: {save_path}")
                save_count += 1
            elif key in [27, ord('q')]:  # ESC或q退出
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()