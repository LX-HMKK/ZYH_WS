#coding=utf-8
"""视觉检测辅助函数（IoU / NMS）。"""
import numpy as np


def calculate_iou(box1, box2):
    """
    计算两个边界框之间的 IoU (Intersection over Union)。

    Args:
        box1, box2: 边界框，格式为 [x1, y1, x2, y2]

    Returns:
        iou: IoU 值
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou


def apply_nms(boxes, classes, scores, iou_threshold=0.5):
    """
    对检测框应用非极大值抑制 (NMS)。

    Args:
        boxes: 检测框列表，每个框格式为 [x1, y1, x2, y2]
        classes: 类别列表
        scores: 置信度分数列表
        iou_threshold: IoU 阈值，默认为 0.5

    Returns:
        indices: 保留下来的检测框索引
    """
    if len(boxes) == 0:
        return []

    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = []

    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)

        if len(sorted_indices) == 1:
            break

        current_box = boxes[current_idx]
        ious = []
        for idx in sorted_indices[1:]:
            iou = calculate_iou(current_box, boxes[idx])
            ious.append(iou)

        ious = np.array(ious)
        remaining_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[1:][remaining_indices]

    return keep_indices
