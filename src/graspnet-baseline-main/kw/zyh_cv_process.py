#通过 “YOLO 检测目标位置→SAM 基于位置精确分割” 的流程，生成两种不同粒度的目标区域掩码，可用于后续的图像分析、目标提取等任务。

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def yolo_and_sam(image_path, sam_mask='mask.png', yolo_bbox_mask='bbox_mask.png'):
    # 加载 YOLO 模型
    yolo_model = YOLO('/home/zyh/zyh_demo1/Yolov8/tool.pt')
    
    # 加载 SAM 模型
    sam_checkpoint = '/home/zyh/zyh_demo1/Yolov8/segment-anything-main/sam_vit_b_01ec64.pth'
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 加载图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # 设置 SAM 图像
    predictor.set_image(image_rgb)

    # YOLO 推理
    results = yolo_model(image, conf=0.3)
    
    # 创建SAM掩码（背景黑，分割白）
    final_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 创建YOLO检测框二值掩码
    bbox_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 遍历检测结果
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        class_label = yolo_model.names[class_id]
        
        # SAM 分割
        input_box = np.array([[x1, y1, x2, y2]])
        masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
        
        if len(masks) > 0:
            mask = masks[0]
            final_mask[mask == 1] = 255  # SAM掩码设为白
        
        # YOLO检测框二值化
        bbox_mask[y1:y2, x1:x2] = 255  # 检测框内区域设为白

    # 保存SAM分割掩码
    cv2.imwrite(sam_mask, final_mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    print(f"SAM掩码已保存到 {sam_mask}")
    
    # 保存YOLO检测框二值图
    cv2.imwrite(yolo_bbox_mask, bbox_mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    print(f"YOLO检测框掩码已保存到 {yolo_bbox_mask}")

if __name__ == '__main__':
    yolo_and_sam(
        '/home/zyh/zyh_demo1/graspnet-baseline-main/kw/aligned_images/color/aligned_color_0000.png',
        sam_mask='sam_mask.png',
        yolo_bbox_mask='yolo_bbox_mask.png'
    )
