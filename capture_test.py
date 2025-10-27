# coding=utf-8
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# --- 配置参数 ---
# 1. YOLOv11 模型加载
# 请替换为您训练好的模型路径
MODEL_PATH = 'D:/StudyWorks/Yolov11/runs/detect/train5/weights/best.pt'
try:
    model = YOLO(MODEL_PATH)
    print(f"成功加载模型: {MODEL_PATH}")
except Exception as e:
    print(f"加载模型失败，请检查模型路径是否正确: {e}")
    # exit() # 为了测试 Realsense，可以先注释掉退出

# 2. RealSense 配置
USE_ROS_BAG = False        # 是否从bag文件读取
ALIGN_TO_COLOR = True      # 将深度流对齐到彩色流
COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
FPS = 30

def get_aligned_frames(pipeline, align):
    """获取对齐后的帧"""
    # 等待新的帧集
    frames = pipeline.wait_for_frames()
    # 对帧进行对齐处理
    aligned_frames = align.process(frames)
    return aligned_frames

def main():
    # 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用流配置
    if USE_ROS_BAG:
        # 如果使用ROS Bag，请修改路径
        config.enable_device_from_file("path/to/your/bag/file.bag")
    else:
        # 启用彩色流 (BGR8 格式，OpenCV常用)
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
        # 启用深度流
        config.enable_stream(rs.stream.depth, COLOR_WIDTH, COLOR_HEIGHT, rs.format.z16, FPS)
    
    # 设置对齐方式：将深度流对齐到彩色流
    align = rs.align(rs.stream.color if ALIGN_TO_COLOR else rs.stream.depth)
    
    # 启动管道
    profile = pipeline.start(config)

    # 设置窗口
    window_name = "RealSense + YOLOv11"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(window_name, COLOR_WIDTH, COLOR_HEIGHT)
    
    try:
        while True:
            # 1. 获取 RealSense 帧
            aligned_frames = get_aligned_frames(pipeline, align)
            
            # 获取彩色帧
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue
                
            # 将彩色帧转换为 OpenCV numpy 数组 (BGR格式)
            color_image = np.asanyarray(color_frame.get_data())
            
            # --- 2. YOLOv11 推理 ---
            
            # 在彩色帧上运行模型推理
            results = model(color_image, stream=True, verbose=False)
            
            # --- 3. 处理并绘制结果 ---
            
            # 遍历每帧的检测结果
            for r in results:
                boxes = r.boxes.cpu().numpy()
                
                # 遍历每个检测到的目标
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # 获取类别名称
                    if model.names:
                        label = f'{model.names[cls]} {conf:.2f}'
                    else:
                        label = f'Class {cls} {conf:.2f}'

                    # 绘制边界框
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制标签文本
                    cv2.putText(
                        color_image, 
                        label, 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2
                    )

            # 4. 显示结果
            cv2.imshow(window_name, color_image)
            
            # 5. 键盘控制
            key = cv2.waitKey(1)
            if key in [27, ord('q')]:  # ESC或q退出
                break
                
    finally:
        # 释放资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense 管道已停止，程序已退出。")

if __name__ == "__main__":
    main()