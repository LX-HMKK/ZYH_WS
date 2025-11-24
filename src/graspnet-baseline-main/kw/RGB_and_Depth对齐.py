#coding=utf-8
import pyrealsense2 as rs
import numpy as np
import cv2
import os  # 新增os模块用于路径处理

# 创建保存目录
output_dir = "aligned_images"
color_dir = os.path.join(output_dir, "color")
depth_dir = os.path.join(output_dir, "depth")
os.makedirs(color_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# 初始化保存计数器
save_count = 0

# 配置参数保持不变
USE_ROS_BAG = 0
ALIGN_WAY = 1

def Align_version(frames, align, show_pic=0):
    # 对齐版本
    aligned_frames = align.process(frames)
    depth_frame_aligned = aligned_frames .get_depth_frame()
    color_frame_aligned = aligned_frames .get_color_frame()
    # if not depth_frame_aligned or not color_frame_aligned:
    #     continue
    color_image_aligned = np.asanyarray(color_frame_aligned.get_data())
    if USE_ROS_BAG:
        color_image_aligned=cv2.cvtColor(color_image_aligned,cv2.COLOR_BGR2RGB)
    depth_image_aligned = np.asanyarray(depth_frame_aligned.get_data())
 
    depth_colormap_aligned = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_aligned, alpha=0.05), cv2.COLORMAP_JET)
    images_aligned = np.hstack((color_image_aligned, depth_colormap_aligned))
    if show_pic:
        cv2.imshow('aligned_images', images_aligned)
    return color_image_aligned,depth_image_aligned,depth_colormap_aligned
 

def Unalign_version(frames, show_pic=0):
   # 未对齐版本
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames .get_depth_frame()
    color_frame = frames .get_color_frame()
 
    if not USE_ROS_BAG:
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)
        left_image = np.asanyarray(left_frame.get_data())
        right_image = np.asanyarray(right_frame.get_data())
        if show_pic:
            cv2.imshow('left_images', left_image)
            cv2.imshow('right_images', right_image)
    # if not depth_frame or not color_frame:
    #     continue
    color_image = np.asanyarray(color_frame.get_data())
    print("color:",color_image.shape)
    depth_image= np.asanyarray(depth_frame.get_data())
    print("depth:",depth_image.shape)
 
    #相机API录制的大小rosbag的rgb图像与depth图像不一致，用resize调整到一样大
    if USE_ROS_BAG:
        color_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
        if ALIGN_WAY:  #深度图对齐到彩色图像
            depth_image=cv2.resize(depth_image,(color_image.shape[1],color_image.shape[0]))
        else:   #彩色图像对齐到深度图
            color_image=cv2.resize(color_image,(depth_image.shape[1],depth_image.shape[0]))
    # 上色
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
    # Stack both images horizontally
    images = np.hstack((color_image, depth_colormap))
    if show_pic:
        cv2.imshow('images', images)
    return color_image,depth_image,depth_colormap
if __name__ == "__main__":
   # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
 
    if USE_ROS_BAG:
        config.enable_device_from_file("666.bag")#这是打开相机API录制的视频
    else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  #10、15或者30可选,20或者25会报错，其他帧率未尝试
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        #左右双目
        config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)  
        config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
 
    if ALIGN_WAY:
        way=rs.stream.color
    else:
        way=rs.stream.depth
    align = rs.align(way)
    profile =pipeline.start(config)
 
 
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("scale:", depth_scale)
    # 深度比例系数为： 0.0010000000474974513

    try:
        while True:
            frames = pipeline.wait_for_frames()
            
            # 处理对齐图像
            color_aligned, depth_aligned, _ = Align_version(frames, align, show_pic=1)
            
            # 处理未对齐图像（保持原有功能）
            color_raw, depth_raw, _ = Unalign_version(frames, show_pic=1)
            
            # 新增保存逻辑
            key = cv2.waitKey(1)
            if key == ord('s'):  # 按s键保存对齐图像
                # 保存彩色图像（自动处理BGR/RGB格式）
                color_path = os.path.join(color_dir, f"aligned_color_{save_count:04d}.png")
                cv2.imwrite(color_path, color_aligned)
                
                # 保存深度图像（强制16位格式）
                depth_path = os.path.join(depth_dir, f"aligned_depth_{save_count:04d}.png")
                cv2.imwrite(depth_path, depth_aligned.astype(np.uint16))
                
                print(f"Saved aligned images: {color_path} & {depth_path}")
                save_count += 1
                
            elif key in [27, ord('q')]:  # ESC或q退出
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()