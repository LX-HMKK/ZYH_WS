#coding=utf-8
import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 创建保存目录
output_dir = "aligned_images"
color_dir = os.path.join(output_dir, "color")
depth_dir = os.path.join(output_dir, "depth")
os.makedirs(color_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# 初始化保存计数器
save_count = 0

# 配置参数
USE_ROS_BAG = 0          # 0:实时相机，1:ROS Bag
ALIGN_WAY = 1            # 1:深度对齐彩色，0:彩色对齐深度
USE_REALSENSE_FILTER = 1 # 1:使用RealSense SDK滤波，0:使用OpenCV滤波
BAG_PATH = "666.bag"     # ROS Bag路径


def init_filters():
    """初始化RealSense深度滤波器（空间+时间+空洞填充）"""
    # 空间滤波：平滑图像同时保留边缘（适合去除高频噪声）
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)       # 滤波强度（1-5，越大越平滑）
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)  # 平滑系数（0-1，越大越平滑）
    spatial.set_option(rs.option.filter_smooth_delta, 20)   # 边缘保留阈值（1-50，越大保留越多边缘）

    # 时间滤波：利用帧间信息减少抖动（适合动态场景）
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)  # 时间平滑系数（0-1，越大越依赖历史帧）
    temporal.set_option(rs.option.filter_smooth_delta, 10)   # 帧间变化阈值（1-100，越小越敏感）

    # 空洞填充：填充深度图中的空洞（因遮挡/反光导致的无效值）
    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 2)  # 填充模式（0:不填充，1:近邻填充，2:多帧融合填充）

    return spatial, temporal, hole_filling


def filter_depth_opencv(depth_img, method='median'):
    """使用OpenCV滤波处理深度图（补充方案）"""
    # 深度图为16位无符号整数，需转换为float处理（避免溢出）
    depth_float = depth_img.astype(np.float32)
    
    if method == 'median':
        # 中值滤波：有效去除椒盐噪声（推荐ksize=3/5）
        filtered = cv2.medianBlur(depth_float, ksize=3)
    elif method == 'bilateral':
        # 双边滤波：平滑噪声同时保留边缘（适合保留物体轮廓）
        filtered = cv2.bilateralFilter(depth_float, d=5, sigmaColor=10, sigmaSpace=10)
    else:
        filtered = depth_float  # 无滤波
    
    # 转回16位无符号整数（深度图标准格式）
    return filtered.astype(np.uint16)


def Align_version(frames, align, filters=None, show_pic=0):
    """对齐版本（含深度滤波）"""
    aligned_frames = align.process(frames)
    depth_frame_aligned = aligned_frames.get_depth_frame()
    color_frame_aligned = aligned_frames.get_color_frame()

    # 彩色图处理
    color_image_aligned = np.asanyarray(color_frame_aligned.get_data())
    if USE_ROS_BAG:
        color_image_aligned = cv2.cvtColor(color_image_aligned, cv2.COLOR_BGR2RGB)

    # 深度图滤波处理
    if USE_REALSENSE_FILTER and filters is not None:
        # 使用RealSense SDK滤波（处理深度帧）
        spatial, temporal, hole_filling = filters
        depth_frame_filtered = spatial.process(depth_frame_aligned)       # 空间滤波
        depth_frame_filtered = temporal.process(depth_frame_filtered)     # 时间滤波
        depth_frame_filtered = hole_filling.process(depth_frame_filtered) # 空洞填充
        depth_image_aligned = np.asanyarray(depth_frame_filtered.get_data())
    else:
        # 使用OpenCV滤波（处理深度数组）
        depth_image_aligned = np.asanyarray(depth_frame_aligned.get_data())
        depth_image_aligned = filter_depth_opencv(depth_image_aligned, method='median')  # 可选'median'/'bilateral'

    # 生成彩色映射图（用于显示）
    depth_colormap_aligned = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image_aligned, alpha=0.05),  # alpha调整亮度
        cv2.COLORMAP_JET
    )
    images_aligned = np.hstack((color_image_aligned, depth_colormap_aligned))
    if show_pic:
        cv2.imshow('Aligned Images (Filtered Depth)', images_aligned)

    return color_image_aligned, depth_image_aligned, depth_colormap_aligned


def Unalign_version(frames, show_pic=0):
    """未对齐版本（保持原有功能）"""
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # 红外图显示（仅实时相机模式）
    if not USE_ROS_BAG:
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)
        left_image = np.asanyarray(left_frame.get_data())
        right_image = np.asanyarray(right_frame.get_data())
        if show_pic:
            cv2.imshow('Left Infrared', left_image)
            cv2.imshow('Right Infrared', right_image)

    # 彩色图与深度图处理
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # ROS Bag尺寸适配
    if USE_ROS_BAG:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        if ALIGN_WAY:
            depth_image = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]))
        else:
            color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]))

    # 深度图彩色映射（用于显示）
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
    images = np.hstack((color_image, depth_colormap))
    if show_pic:
        cv2.imshow('Raw Images (Unfiltered)', images)

    return color_image, depth_image, depth_colormap


if __name__ == "__main__":
    # 初始化相机管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置输入源（实时相机/ROS Bag）
    if USE_ROS_BAG:
        if not os.path.exists(BAG_PATH):
            raise FileNotFoundError(f"ROS Bag文件不存在：{BAG_PATH}")
        config.enable_device_from_file(BAG_PATH)
        print(f"已加载ROS Bag：{BAG_PATH}")
    else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
        print("已启动RealSense实时相机")

    # 初始化对齐器
    align_to = rs.stream.color if ALIGN_WAY else rs.stream.depth
    align = rs.align(align_to)

    # 启动相机并初始化滤波器
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例系数：{depth_scale:.6f} 米/像素")
    filters = init_filters() if USE_REALSENSE_FILTER else None  # 初始化滤波器

    try:
        while True:
            frames = pipeline.wait_for_frames()
            
            # 处理对齐图像（含滤波）
            color_aligned, depth_aligned, _ = Align_version(
                frames, align, filters=filters, show_pic=1
            )
            
            # 处理未对齐图像（原始数据，用于对比）
            color_raw, depth_raw, _ = Unalign_version(frames, show_pic=1)
            
            # 按键交互（保存/退出）
            key = cv2.waitKey(1)
            if key == ord('s'):  # 按s键保存滤波后的对齐图像
                color_path = os.path.join(color_dir, f"aligned_color_{save_count:04d}.png")
                depth_path = os.path.join(depth_dir, f"aligned_depth_{save_count:04d}.png")
                
                # 保存彩色图（BGR格式，适配OpenCV）
                cv2.imwrite(color_path, color_aligned)
                # 保存滤波后的深度图（16位格式，保留原始深度信息）
                cv2.imwrite(depth_path, depth_aligned.astype(np.uint16))
                
                print(f"已保存滤波后的图像：\n  彩色图：{color_path}\n  深度图：{depth_path}")
                save_count += 1
                
            elif key in [27, ord('q')]:  # ESC或q退出
                print("程序退出")
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
