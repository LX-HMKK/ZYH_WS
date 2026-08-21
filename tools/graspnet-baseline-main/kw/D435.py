import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 配置相机管道
pipeline = rs.pipeline()  # 创建管道，用于连接相机和获取数据
config = rs.config()      # 创建配置对象，设置要启用的流

# 配置流：彩色图（640x480，30fps，RGB格式）
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# 配置流：深度图（640x480，30fps，16位深度值）
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 2. 启动管道（开始获取数据）
profile = pipeline.start(config)

try:
    while True:
        # 3. 获取一帧数据（等待新帧）
        frames = pipeline.wait_for_frames()
        
        # 分离彩色帧和深度帧
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        # 检查帧是否有效
        if not color_frame or not depth_frame:
            continue
        
        # 4. 转换为 numpy 数组（便于处理和显示）
        color_image = np.asanyarray(color_frame.get_data())  # 彩色图（BGR格式，OpenCV默认）
        depth_image = np.asanyarray(depth_frame.get_data())  # 深度图（16位整数，单位：毫米）
        
        # 5. 深度图彩色映射（便于可视化，可选）
        # 将深度值映射到彩色范围（0-255），使用Jet配色方案
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # 6. 显示图像
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image (Colormap)', depth_colormap)
        
        # 按 'q' 退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 7. 停止管道，释放资源
    pipeline.stop()
    cv2.destroyAllWindows()
