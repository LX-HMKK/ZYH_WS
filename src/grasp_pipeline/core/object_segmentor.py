#coding=utf-8
"""YOLO 检测 + SAM 分割，生成目标掩码。"""
import os
import numpy as np
import cv2

from ..config import Config
from ..utils.vision_utils import apply_nms


class ObjectSegmentor:
    """基于 YOLO 检测与 SAM 分割生成目标掩码。"""

    def __init__(self, yolo_model, sam_predictor, device: str, config: Config | None = None):
        self.yolo_model = yolo_model
        self.sam_predictor = sam_predictor
        self.device = device
        self.config = config or Config

    def generate_masks(self, color_img: np.ndarray, color_save_path: str, interactive: bool = False):
        """
        基于对齐彩色图生成 SAM 分割掩码和 YOLO 扩展掩码。

        Args:
            color_img: 对齐后的 RGB 彩色图
            color_save_path: 彩色图保存路径（用于生成掩码文件名）
            interactive: 是否显示中间结果窗口

        Returns:
            (sam_mask_path, yolo_mask_path, cls_name)
        """
        height, width = color_img.shape[:2]
        color_rgb = color_img
        self.sam_predictor.set_image(color_rgb)

        # YOLO 检测：interactive 用 0.3，自动模式用 0.7
        conf = 0.3 if interactive else 0.7
        results = self.yolo_model(color_img, conf=conf, device=self.device)

        if len(results[0].boxes) == 0:
            print("警告：未检测到任何目标，生成全黑掩码")
            sam_mask = np.zeros((height, width), dtype=np.uint8)
            yolo_mask = np.zeros((height, width), dtype=np.uint8)
            cls_name = "unknown"
        else:
            sam_mask = np.zeros((height, width), dtype=np.uint8)
            yolo_mask = np.zeros((height, width), dtype=np.uint8)

            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()

            nms_indices = apply_nms(boxes, classes, scores, iou_threshold=0.5)

            # 按检测框面积排序（从大到小）
            indexed_boxes = [
                (idx, (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1]))
                for idx in nms_indices
            ]
            sorted_indices = [idx for idx, _ in sorted(indexed_boxes, key=lambda x: x[1], reverse=True)]

            cls_name = "unknown"
            if sorted_indices:
                idx = sorted_indices[0]
                box = boxes[idx]
                cls = classes[idx]
                score = scores[idx]

                x1, y1, x2, y2 = map(int, box)
                cls_name = self.yolo_model.names[int(cls)]
                print(f"检测到目标：{cls_name}，置信度：{score:.2f}，坐标：({x1},{y1})-({x2},{y2})")

                # SAM 分割（基于 YOLO 检测框）
                input_box = np.array([[x1, y1, x2, y2]])
                masks, _, _ = self.sam_predictor.predict(box=input_box, multimask_output=False)
                if len(masks) > 0:
                    kernel = np.ones((30, 30), np.uint8)
                    expanded_sam_mask = cv2.morphologyEx(masks[0].astype(np.uint8), cv2.MORPH_DILATE, kernel)
                    sam_mask[expanded_sam_mask == 1] = 255

                # YOLO 扩展掩码
                x1_exp = max(0, x1)
                y1_exp = max(0, y1)
                x2_exp = min(width, x2)
                y2_exp = min(height, y2)
                yolo_mask[y1_exp:y2_exp, x1_exp:x2_exp] = 255

        # 保存掩码
        base_name = os.path.splitext(os.path.basename(color_save_path))[0]
        sam_mask_path = os.path.join(self.config.MASK_SAVE_DIR, f"{base_name}_sam_mask.png")
        yolo_mask_path = os.path.join(self.config.MASK_SAVE_DIR, f"{base_name}_yolo_mask.png")

        cv2.imwrite(sam_mask_path, sam_mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        cv2.imwrite(yolo_mask_path, yolo_mask, [cv2.IMWRITE_PNG_BILEVEL, 1])

        print(f"已保存掩码：\n  - SAM分割掩码：{sam_mask_path}\n  - YOLO扩展掩码：{yolo_mask_path}")

        if interactive:
            self._show_sam_result(color_img, sam_mask)

        return sam_mask_path, yolo_mask_path, cls_name

    def generate_masks_auto(self, color_img: np.ndarray, color_save_path: str):
        """自动模式生成掩码（不显示窗口）。"""
        return self.generate_masks(color_img, color_save_path, interactive=False)

    def _show_sam_result(self, color_img: np.ndarray, sam_mask: np.ndarray):
        """显示 SAM 分割结果（交互模式专用）。"""
        display_img = color_img.copy()
        sam_mask_3channel = cv2.cvtColor(sam_mask, cv2.COLOR_GRAY2BGR)
        green_mask = np.zeros_like(display_img)
        green_mask[:, :] = [0, 255, 0]
        masked_area = cv2.bitwise_and(green_mask, sam_mask_3channel)
        cv2.addWeighted(display_img, 0.7, masked_area, 0.3, 0, display_img)

        display_height = 480
        display_width = int(display_img.shape[1] * display_height / display_img.shape[0])
        display_img_resized = cv2.resize(display_img, (display_width, display_height))

        cv2.imshow('SAM Segmentation Result', display_img_resized)
        print("按任意键关闭SAM分割显示窗口...")
        cv2.waitKey(0)
        cv2.destroyWindow('SAM Segmentation Result')

        mask_display = cv2.resize(sam_mask, (display_width, display_height))
        cv2.imshow('SAM Binary Mask', mask_display)
        print("按任意键关闭二值掩码显示窗口...")
        cv2.waitKey(0)
        cv2.destroyWindow('SAM Binary Mask')
