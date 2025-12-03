#coding=utf-8
import pyrealsense2 as rs
import numpy as np
import cv2
import os  # 新增os模块用于路径处理
from ultralytics import YOLO
import logging

# 禁用 Ultralytics 日志输出
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# 创建保存目录
output_dir = "aligned_images"
color_dir = os.path.join(output_dir, "color")
os.makedirs(color_dir, exist_ok=True)

# 初始化保存计数器
save_count = 0

# 加载 YOLO 模型
yolo_model = YOLO('/home/zyh/ZYH_WS/src/graspnet-baseline-main/all.pt')

USE_ROS_BAG = 0

def calculate_iou(box1, box2):
    """
    计算两个边界框之间的IOU
    box: [x1, y1, x2, y2]
    """
    # 计算交集坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算各自边界框面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = box1_area + box2_area - inter_area
    
    # 计算IOU
    if union_area == 0:
        return 0
    iou = inter_area / union_area
    return iou


def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    对检测框应用非极大值抑制
    boxes: 检测框列表 [[x1, y1, x2, y2], ...]
    scores: 置信度分数列表
    iou_threshold: IOU阈值
    """
    if len(boxes) == 0:
        return []
    
    # 转换为numpy数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 使用opencv的NMSBoxes函数
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou_threshold)
    
    if len(indices) > 0:
        # 展平索引数组 (opencv返回的格式可能为[[i], [j]]或[i, j])
        if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
            indices = [idx[0] for idx in indices]
        return indices
    else:
        return []

def process_yolo_detection(color_image):
    """处理YOLO检测并在图像上绘制结果"""
    # YOLO检测
    results = yolo_model(color_image, conf=0.7,verbose=False)
    
    # 复制图像用于显示
    display_image = color_image.copy()
    curr_boxes = []
    curr_scores = []
    detections = []
    
    # 遍历检测结果
    for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        confidence = float(conf)
        class_label = yolo_model.names[class_id]
        
        # 保存检测信息
        curr_boxes.append([x1, y1, x2, y2])
        curr_scores.append(confidence)
        detections.append({
            'box': [x1, y1, x2, y2],
            'class_id': class_id,
            'confidence': confidence,
            'label': class_label
        })
    
    # 应用非极大值抑制
    nms_indices = apply_nms(curr_boxes, curr_scores, iou_threshold=0.5)
    
    # 只绘制NMS后的检测框
    filtered_detections = []
    filtered_objects = []  # 存储NMS后的类名和置信度
    if len(nms_indices) > 0:
        for i in nms_indices:
            detection = detections[i]
            x1, y1, x2, y2 = detection['box']
            class_label = detection['label']
            confidence = detection['confidence']
            
            # 绘制检测框
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_image, f'{class_label} {confidence:.2f}', 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            filtered_detections.append(detection['box'])
            filtered_objects.append((class_label, confidence))  # 添加过滤后的类名和置信度
    
     # 打印检测到的类名和置信度
    if filtered_objects:
        object_info = [f"{label} ({confidence:.2f})" for label, confidence in filtered_objects]
        print(f"检测到的物体: {', '.join(object_info)}")
    # else:
        # print("未检测到任何物体")


    # 计算IOU（如果存在前一帧的检测框）
    if hasattr(process_yolo_detection, 'prev_boxes') and process_yolo_detection.prev_boxes:
        iou_values = []
        for prev_box in process_yolo_detection.prev_boxes:
            max_iou = 0
            for curr_box in filtered_detections:
                iou = calculate_iou(prev_box, curr_box)
                max_iou = max(max_iou, iou)
            if max_iou > 0:
                iou_values.append(max_iou)
        
        if iou_values:
            avg_iou = np.mean(iou_values)
            cv2.putText(display_image, f'Avg IOU: {avg_iou:.2f}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 保存当前帧的检测框
    process_yolo_detection.prev_boxes = filtered_detections
    
    return display_image

def RGB_version(frames, show_pic=0):
    """仅处理RGB图像版本"""
    # 获取彩色帧
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        return None, None
    
    # 将彩色帧转换为numpy数组
    color_image = np.asanyarray(color_frame.get_data())
    
    if USE_ROS_BAG:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # 在RGB图像上进行YOLO检测
    color_with_detection = process_yolo_detection(color_image)
    
    if show_pic:
        cv2.imshow('rgb_images', color_with_detection)
    
    return color_image, color_with_detection

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    if USE_ROS_BAG:
        config.enable_device_from_file("666.bag")#这是打开相机API录制的视频
    else:
        # 只启用RGB流
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    # 启动管道
    profile = pipeline.start(config)
    
    # 设置超时时间（毫秒）
    timeout_count = 0
    max_timeout_count = 3
    
    try:
        while True:
            # 等待一帧数据，设置超时时间为5秒
            try:
                frames = pipeline.wait_for_frames(timeout_ms=3000)
                timeout_count = 0  # 重置超时计数
            except RuntimeError as e:
                print(f"等待帧超时: {e}")
                timeout_count += 1
                if timeout_count >= max_timeout_count:
                    print("连续超时次数过多，退出程序")
                    break
                continue
            
            # 处理RGB图像
            color_raw, color_detected = RGB_version(frames, show_pic=1)
            
            # 新增保存逻辑
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # 按s键保存对齐图像
                # 保存彩色图像（自动处理BGR/RGB格式）
                color_path = os.path.join(color_dir, f"color_{save_count:04d}.png")
                cv2.imwrite(color_path, color_raw)
                
                print(f"Saved RGB image: {color_path}")
                save_count += 1
                
            elif key in [27, ord('q')]:  # ESC或q退出
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()