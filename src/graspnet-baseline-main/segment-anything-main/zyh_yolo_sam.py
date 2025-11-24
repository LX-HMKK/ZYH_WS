import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# 加载 YOLO 模型
model = YOLO('/home/zyh/zyh_demo1/Yolov8/tool.pt') 

# 加载 SAM 模型
sam_checkpoint = '/home/zyh/zyh_demo1/Yolov8/segment-anything-main/sam_vit_b_01ec64.pth'  # 替换为你的模型路径
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 加载图像
image_path ='/home/zyh/zyh_demo1/Yolov8/tool.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式以供 SAM 使用

# 运行 YOLO 推理
results = model(image, conf=0.3)

# 遍历检测结果
for result in results:
    # 获取边界框
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):   
        x1, y1, x2, y2 = map(int, box)  # 转换为整数
        
        # 获取 ID
        class_id = int(cls)  # 类别 ID
        # 获取类别标签
        class_label = model.names[class_id]   
        # 准备 SAM
        predictor.set_image(image_rgb)

        # 定义一个边界框提示供 SAM 使用
        input_box = np.array([[x1, y1, x2, y2]])

        # 获取 SAM 掩码
        masks, _, _ = predictor.predict(box=input_box, multimask_output=False)

        # 创建原始图像的副本以叠加掩码
        highlighted_image = image_rgb.copy()

        # 将掩码以半透明蓝色应用于图像
        mask = masks[0]
        # 创建一个空白图像
        blue_overlay = np.zeros_like(image_rgb, dtype=np.uint8)  
        # 蓝色用于分割区域（RGB）
        blue_overlay[mask == 1] = [0, 0, 255]   

        # 使用透明度将蓝色叠加层与原始图像混合
        alpha = 0.7  # 叠加层的透明度
        highlighted_image = cv2.addWeighted(highlighted_image, 1 - alpha, blue_overlay, alpha, 0)

         # 在边界框上方添加标签（类别名称）
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"{class_label}"  # 标签为类别名称
        cv2.putText(highlighted_image, label, (x1, y1 - 10), font, 2, (255, 255, 0), 2, cv2.LINE_AA)   

        # 可选：保存带有边界框和突出显示的分割结果的图像
        output_filename = f"highlighted_output.png"
        cv2.imwrite(output_filename, cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))

