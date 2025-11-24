#coding=utf-8
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import open3d as o3d
from graspnetAPI import GraspGroup
# 新增：坐标转换依赖库
from scipy.spatial.transform import Rotation as R

# -------------------------- 第一步：整合依赖路径与全局配置（新增机械臂参数） --------------------------
# 1. 配置GraspNet依赖路径（根据实际项目结构调整）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, 'models'))    # GraspNet模型目录
sys.path.append(os.path.join(PARENT_DIR, 'dataset'))   # 数据处理目录
sys.path.append(os.path.join(PARENT_DIR, 'utils'))     # 工具函数目录

# 2. 导入GraspNet相关模块
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# 3. 全局参数配置（统一管理：相机/模型/机械臂手眼参数）
class Config:
    # ---------------------- 原有参数 ----------------------
    # 相机与对齐配置
    USE_ROS_BAG = 0                  # 0:实时相机，1:ROS Bag
    ALIGN_WAY = 1                    # 1:深度对齐彩色，0:彩色对齐深度
    BAG_PATH = "666.bag"             # ROS Bag路径（USE_ROS_BAG=1时生效）
    CAMERA_RES = (1280, 720)         # RealSense分辨率（D435/D435i默认）
    CAMERA_FPS = 30                  # 帧率
    
    # 模型路径（请根据实际路径修改！）
    YOLO_MODEL_PATH = '/home/zyh/zyh_demo1/graspnet-baseline-main/all.pt'
    SAM_CHECKPOINT = '/home/zyh/zyh_demo1/Yolov8/segment-anything-main/sam_vit_b_01ec64.pth'
    SAM_MODEL_TYPE = "vit_b"
    GRASP_CHECKPOINT = '/home/zyh/zyh_demo1/graspnet-baseline-main/logs/log_kn/checkpoint.tar'
    
    # 输出目录
    OUTPUT_ROOT = "aligned_images"
    COLOR_SAVE_DIR = os.path.join(OUTPUT_ROOT, "color")
    DEPTH_SAVE_DIR = os.path.join(OUTPUT_ROOT, "depth")
    MASK_SAVE_DIR = os.path.join(OUTPUT_ROOT, "masks")
    
    # GraspNet抓取参数
    NUM_POINT = 10000                # 点云采样数量
    NUM_VIEW = 300                   # 候选抓取视角数
    COLLISION_THRESH = 0.001         # 碰撞检测阈值（米）
    VOXEL_SIZE = 0.001                # 点云体素化分辨率（米）
    DEPTH_FACTOR = 1000.0            # 深度因子（D435默认1000）
    # D435相机内参（需与实际校准匹配）
    DEPTH_INTR = {
        "ppx": 644.136,  # 主点x
        "ppy": 354.556,  # 主点y
        "fx": 902.806,   # 焦距x
        "fy": 900.776    # 焦距y
    }
    MASK_CHOICE =   0        # 0:SAM掩码，1:YOLO扩展掩码
    
    # ---------------------- 新增：机械臂与手眼标定参数 ----------------------
    # 1. 机械臂当前末端位姿（基座坐标系下，格式：[x, y, z, rx, ry, rz]，需根据实际机械臂状态修改）
    CURRENT_EE_POSE = [
        0.10809402105592679,   # x (m)
        0.3258393975737154,    # y (m)
        -0.09185500000868108,  # z (m)
        2.5413909196778034e-26,# rx (rad, XYZ欧拉角)
        -1.2627174896584495e-05,# ry (rad)
        3.1415686851497933     # rz (rad)
    ]
    
    # 2. 手眼标定参数（末端→相机的变换：旋转矩阵+平移向量，需根据实际标定结果修改）
    HANDEYE_ROT = np.array([
        [ 0.02840914 , 0.99959434, -0.00201865],
        [-0.99937999 , 0.02844495,  0.02074905],
        [ 0.02079805 , 0.00142794, 0.99978268]
    ])
    HANDEYE_TRANS = np.array([  # 末端→相机的平移向量（m）
     -0.09387686988,
      0.02960431733,
     -0.21970771816
    ])


# 初始化输出目录
os.makedirs(Config.COLOR_SAVE_DIR, exist_ok=True)
os.makedirs(Config.DEPTH_SAVE_DIR, exist_ok=True)
os.makedirs(Config.MASK_SAVE_DIR, exist_ok=True)


def calculate_iou(box1, box2):
    """
    计算两个边界框之间的IoU (Intersection over Union)
    
    Args:
        box1, box2: 边界框，格式为 [x1, y1, x2, y2]
    
    Returns:
        iou: IoU值
    """
    # 计算交集区域坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # 计算各自框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = area1 + area2 - inter_area
    
    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def apply_nms(boxes, classes, scores, iou_threshold=0.5):
    """
    对检测框应用非极大值抑制(NMS)
    
    Args:
        boxes: 检测框列表，每个框格式为 [x1, y1, x2, y2]
        classes: 类别列表
        scores: 置信度分数列表
        iou_threshold: IoU阈值，默认为0.5
    
    Returns:
        indices: 保留下来的检测框索引
    """
    if len(boxes) == 0:
        return []
    
    # 按照置信度分数降序排列索引
    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # 保留当前置信度最高的检测框
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
            
        # 计算当前框与其他所有框的IoU
        ious = []
        current_box = boxes[current_idx]
        for idx in sorted_indices[1:]:
            iou = calculate_iou(current_box, boxes[idx])
            ious.append(iou)
        
        # 保留IoU小于阈值的框（即不重叠过多的框）
        ious = np.array(ious)
        remaining_indices = np.where(ious < iou_threshold)[0]
        
        # 更新sorted_indices为剩余的框
        sorted_indices = sorted_indices[1:][remaining_indices]
    
    return keep_indices

def convert_grasp_to_robot_base(
        grasp_translation,  # GraspNet输出：相机坐标系下的抓取平移 (m)
        grasp_rotation_mat,  # GraspNet输出：相机坐标系下的抓取旋转矩阵 (3x3)
        current_ee_pose=Config.CURRENT_EE_POSE,  # 机械臂当前末端位姿
        handeye_rot=Config.HANDEYE_ROT,          # 手眼标定：末端→相机旋转矩阵
        handeye_trans=Config.HANDEYE_TRANS       # 手眼标定：末端→相机平移向量
):
    # 坐标系对齐矩阵
    R_adjust = np.array([
        [0, 0, 1],  # X轴旋转到Z轴
        [0, 1, 0],  # Y轴保持不动
        [-1, 0, 0]  # Z轴旋转到-X轴
    ], dtype=np.float32)
    
    T_align = np.eye(4, dtype=float)
    T_align[:3, :3] = R_adjust
    
    # 抓取位姿到相机的变换矩阵（应用坐标系对齐）
    T_grasp2cam = np.eye(4)
    T_grasp2cam[:3, :3] = grasp_rotation_mat
    T_grasp2cam[:3, 3] = grasp_translation
    T_grasp2cam = T_grasp2cam @ T_align

    # 手眼标定矩阵（相机→末端）
    T_cam2end = np.eye(4)
    T_cam2end[:3, :3] = handeye_rot
    T_cam2end[:3, 3] = handeye_trans

    # 末端到基座变换
    x, y, z, rx, ry, rz = current_ee_pose
    R_end2base = R.from_euler('XYZ', [rx, ry, rz]).as_matrix()
    T_end2base = np.eye(4)
    T_end2base[:3, :3] = R_end2base
    T_end2base[:3, 3] = [x, y, z]

    # 基座到抓取的完整变换链
    T_base2grasp = T_end2base @ T_cam2end @ T_grasp2cam
    
    # 提取最终位姿
    final_trans = T_base2grasp[:3, 3]
    final_rot = R.from_matrix(T_base2grasp[:3, :3])
    base_rx, base_ry, base_rz = final_rot.as_euler('XYZ')
    
    return np.concatenate([final_trans, [base_rx, base_ry, base_rz]])

# -------------------------- 第三步：加载所有模型（原逻辑保留） --------------------------
def load_all_models():
    """加载YOLO、SAM、GraspNet模型，返回模型实例"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 模型加载（设备：{device}）===")
    
    # 1. 加载YOLO检测模型
    print("1. 加载YOLO模型...")
    yolo_model = YOLO(Config.YOLO_MODEL_PATH)
    yolo_model.to(device)
    
    # 2. 加载SAM分割模型
    print("2. 加载SAM模型...")
    sam = sam_model_registry[Config.SAM_MODEL_TYPE](checkpoint=Config.SAM_CHECKPOINT)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    # 3. 加载GraspNet抓取模型
    print("3. 加载GraspNet模型...")
    grasp_net = GraspNet(
        input_feature_dim=0,
        num_view=Config.NUM_VIEW,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    grasp_net.to(device)
    checkpoint = torch.load(Config.GRASP_CHECKPOINT, map_location=device)
    grasp_net.load_state_dict(checkpoint['model_state_dict'])
    grasp_net.eval()  # 切换为评估模式
    
    print("=== 所有模型加载完成 ===\n")
    return yolo_model, sam_predictor, grasp_net, device


# -------------------------- 第四步：图像对齐与基础处理（原逻辑保留） --------------------------
def process_aligned_frames(frames, aligner, use_bag):
    """处理对齐的彩色图与深度图，返回：对齐彩色图（RGB）、对齐深度图（16位）、深度彩色映射图"""
    aligned_frames = aligner.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    # 转换彩色图（BGR→RGB，适配后续模型）
    color_img = np.asanyarray(color_frame.get_data())
    if use_bag:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    
    # 转换深度图（16位格式，保留原始深度）
    depth_img = np.asanyarray(depth_frame.get_data())
    
    # 生成深度彩色映射图（用于显示）
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_img, alpha=0.05), 
        cv2.COLORMAP_JET
    )
    
    return color_img, depth_img, depth_colormap


def process_unaligned_frames(frames, use_bag, align_way):
    """处理未对齐的原始帧（含红外图显示），返回原始彩色图、深度图"""
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())
    
    # 适配ROS Bag尺寸不一致问题
    if use_bag:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        if align_way:
            depth_img = cv2.resize(depth_img, (color_img.shape[1], color_img.shape[0]))
        else:
            color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]))
    return color_img, depth_img


# -------------------------- 第五步：YOLO+SAM掩码生成（原逻辑保留） --------------------------
def generate_masks(color_img, color_save_path, yolo_model, sam_predictor, device):
    """基于对齐彩色图生成SAM分割掩码和YOLO扩展掩码，返回掩码路径"""
    height, width = color_img.shape[:2]
    color_rgb = color_img  # 已在对齐阶段转换为RGB
    sam_predictor.set_image(color_rgb)  # SAM设置输入图像
    
    # YOLO检测（置信度0.3）
    results = yolo_model(color_img, conf=0.3, device=device)
    if len(results[0].boxes) == 0:
        print("警告：未检测到任何目标，生成全黑掩码")
        sam_mask = np.zeros((height, width), dtype=np.uint8)
        yolo_mask = np.zeros((height, width), dtype=np.uint8)
    else:
        sam_mask = np.zeros((height, width), dtype=np.uint8)
        yolo_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 提取检测框、类别和置信度分数
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        
        # 应用NMS
        nms_indices = apply_nms(boxes, classes, scores, iou_threshold=0.5)

        # 按检测框面积排序（从大到小）
        indexed_boxes = [(idx, (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])) for idx in nms_indices]
        sorted_indices = [idx for idx, area in sorted(indexed_boxes, key=lambda x: x[1], reverse=True)]
        
        
        # 遍历通过NMS筛选后的检测目标生成掩码
        # for idx in sorted_indices:
        if sorted_indices:
            idx = sorted_indices[0]
            box = boxes[idx]
            cls = classes[idx]
            score = scores[idx]
            
            x1, y1, x2, y2 = map(int, box)
            cls_name = yolo_model.names[int(cls)]
            print(f"检测到目标：{cls_name}，置信度：{score:.2f}，坐标：({x1},{y1})-({x2},{y2})")
            
            # SAM分割（基于YOLO检测框）
            input_box = np.array([[x1, y1, x2, y2]])
            masks, _, _ = sam_predictor.predict(box=input_box, multimask_output=False)
            if len(masks) > 0:
                kernel = np.ones((30,30), np.uint8)  # 调整核大小控制扩展范围
                expanded_sam_mask = cv2.morphologyEx(masks[0].astype(np.uint8), cv2.MORPH_DILATE, kernel)
                sam_mask[expanded_sam_mask == 1] = 255
            
            # YOLO扩展掩码（扩展150像素，避免边界截断）
            x1_exp = max(0, x1)
            y1_exp = max(0, y1 )
            x2_exp = min(width, x2 )
            y2_exp = min(height, y2 )
            yolo_mask[y1_exp:y2_exp, x1_exp:x2_exp] = 255
    
    # 保存掩码
    base_name = os.path.splitext(os.path.basename(color_save_path))[0]
    sam_mask_path = os.path.join(Config.MASK_SAVE_DIR, f"{base_name}_sam_mask.png")
    yolo_mask_path = os.path.join(Config.MASK_SAVE_DIR, f"{base_name}_yolo_mask.png")
    
    cv2.imwrite(sam_mask_path, sam_mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    cv2.imwrite(yolo_mask_path, yolo_mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    
    print(f"已保存掩码：\n  - SAM分割掩码：{sam_mask_path}\n  - YOLO扩展掩码：{yolo_mask_path}")
    
    # 显示SAM分割结果
    # 创建彩色图像用于显示（原始图像+掩码叠加）
    display_img = color_img.copy()
    # 将掩码转换为3通道以便可视化
    sam_mask_3channel = cv2.cvtColor(sam_mask, cv2.COLOR_GRAY2BGR)
    # 创建半透明的绿色掩码层
    green_mask = np.zeros_like(display_img)
    green_mask[:, :] = [0, 255, 0]  # 绿色
    # 应用掩码
    masked_area = cv2.bitwise_and(green_mask, sam_mask_3channel)
    # 将掩码叠加到原始图像上（半透明效果）
    cv2.addWeighted(display_img, 0.7, masked_area, 0.3, 0, display_img)
    
    # 调整显示尺寸以便查看
    display_height = 480
    display_width = int(display_img.shape[1] * display_height / display_img.shape[0])
    display_img_resized = cv2.resize(display_img, (display_width, display_height))
    
    # 显示图像
    cv2.imshow('SAM Segmentation Result', display_img_resized)
    print("按任意键关闭SAM分割显示窗口...")
    cv2.waitKey(0)
    cv2.destroyWindow('SAM Segmentation Result')
    
    # 同时显示二值化掩码
    mask_display = cv2.resize(sam_mask, (display_width, display_height))
    cv2.imshow('SAM Binary Mask', mask_display)
    print("按任意键关闭二值掩码显示窗口...")
    cv2.waitKey(0)
    cv2.destroyWindow('SAM Binary Mask')
    
    return sam_mask_path, yolo_mask_path, cls_name
##################################################################################################

# 自动版本
def generate_masks_auto(color_img, color_save_path, yolo_model, sam_predictor, device):
    """基于对齐彩色图生成SAM分割掩码和YOLO扩展掩码，返回掩码路径（自动模式，无用户交互）"""
    height, width = color_img.shape[:2]
    color_rgb = color_img  # 已在对齐阶段转换为RGB
    sam_predictor.set_image(color_rgb)  # SAM设置输入图像
    
    # YOLO检测（置信度0.7）
    results = yolo_model(color_img, conf=0.7, device=device)
    if len(results[0].boxes) == 0:
        print("警告：未检测到任何目标，生成全黑掩码")
        sam_mask = np.zeros((height, width), dtype=np.uint8)
        yolo_mask = np.zeros((height, width), dtype=np.uint8)
    else:
        sam_mask = np.zeros((height, width), dtype=np.uint8)
        yolo_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 提取检测框、类别和置信度分数
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        
        # 应用NMS
        nms_indices = apply_nms(boxes, classes, scores, iou_threshold=0.5)

        # 按检测框面积排序（从大到小）
        indexed_boxes = [(idx, (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])) for idx in nms_indices]
        sorted_indices = [idx for idx, area in sorted(indexed_boxes, key=lambda x: x[1], reverse=True)]
        
        
        # 遍历通过NMS筛选后的检测目标生成掩码
        # for idx in sorted_indices:
        if sorted_indices:
            idx = sorted_indices[0]
            box = boxes[idx]
            cls = classes[idx]
            score = scores[idx]
            
            x1, y1, x2, y2 = map(int, box)
            cls_name = yolo_model.names[int(cls)]
            print(f"检测到目标：{cls_name}，置信度：{score:.2f}，坐标：({x1},{y1})-({x2},{y2})")
            
            # SAM分割（基于YOLO检测框）
            input_box = np.array([[x1, y1, x2, y2]])
            masks, _, _ = sam_predictor.predict(box=input_box, multimask_output=False)
            if len(masks) > 0:
                kernel = np.ones((30,30), np.uint8)  # 调整核大小控制扩展范围
                expanded_sam_mask = cv2.morphologyEx(masks[0].astype(np.uint8), cv2.MORPH_DILATE, kernel)
                sam_mask[expanded_sam_mask == 1] = 255
            
            # YOLO扩展掩码（扩展150像素，避免边界截断）
            x1_exp = max(0, x1)
            y1_exp = max(0, y1 )
            x2_exp = min(width, x2 )
            y2_exp = min(height, y2 )
            yolo_mask[y1_exp:y2_exp, x1_exp:x2_exp] = 255
    
    # 保存掩码
    base_name = os.path.splitext(os.path.basename(color_save_path))[0]
    sam_mask_path = os.path.join(Config.MASK_SAVE_DIR, f"{base_name}_sam_mask.png")
    yolo_mask_path = os.path.join(Config.MASK_SAVE_DIR, f"{base_name}_yolo_mask.png")
    
    cv2.imwrite(sam_mask_path, sam_mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    cv2.imwrite(yolo_mask_path, yolo_mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    
    print(f"已保存掩码：\n  - SAM分割掩码：{sam_mask_path}\n  - YOLO扩展掩码：{yolo_mask_path}")
    
    return sam_mask_path, yolo_mask_path, cls_name


def run_grasp_prediction(grasp_net, color_path, depth_path, mask_path):
    """执行抓取位姿预测，并调用坐标转换函数，返回：最优抓取位姿+基座坐标系位姿"""
    print("\n=== 开始抓取位姿预测 ===")
    device = next(grasp_net.parameters()).device
    
    # 1. 处理输入数据（彩色图+深度图+掩码→点云）
    end_points, cloud_o3d = get_and_process_grasp_data(color_path, depth_path, mask_path)
    
    # 获取点云数据用于计算物体最低点
    point_cloud_points = np.asarray(cloud_o3d.points)
    if len(point_cloud_points) > 0:
        # 物体最低点（最小z坐标）
        object_min_z = np.min(point_cloud_points[:, 2])
        print(f"物体最低点高度: {object_min_z:.6f}m")
    else:
        object_min_z = None
    
    # 获取YOLO检测框中心点
    color_img = np.array(Image.open(color_path))
    yolo_model = YOLO(Config.YOLO_MODEL_PATH)
    results = yolo_model(color_img, conf=0.3)
    
    target_centers = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            target_centers.append((center_x, center_y))
    
    # 2. GraspNet前向推理
    with torch.no_grad():
        end_points = grasp_net(end_points)
        grasp_preds = pred_decode(end_points)
    
    # 3. 构建原始抓取组并检查是否为空
    grasp_group = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    if len(grasp_group) == 0:
        print("错误：GraspNet未预测到任何抓取位姿！请检查输入点云或模型权重。")
        return None  # 提前返回，避免后续错误
    
    # 4. 碰撞检测（并检查结果）
    if Config.COLLISION_THRESH > 0:
        grasp_group = collision_detection(grasp_group, np.asarray(cloud_o3d.points))
        if len(grasp_group) == 0:
            print("警告：所有抓取位姿均与背景碰撞！尝试降低 COLLISION_THRESH 参数。")
            # 若碰撞检测后为空，使用原始抓取组重试
            grasp_group = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    
    # 5. NMS去重
    grasp_group.nms()
    
    # 6. 根据夹爪夹角大小、距离目标中心的远近和高度约束排序
    print("=== 根据夹爪夹角大小、距离目标中心和高度约束排序 ===")
    vertical_dir = np.array([0, 0, 1])
    
    # 获取相机内参
    fx = Config.DEPTH_INTR['fx']
    fy = Config.DEPTH_INTR['fy']
    cx = Config.DEPTH_INTR['ppx']
    cy = Config.DEPTH_INTR['ppy']
    
    # 计算每个抓取与垂直方向的夹角，并存储抓取和角度信息
    grasp_with_info = []
    valid_grasp_count = 0
    
    for grasp in grasp_group:
        approach_dir = grasp.rotation_matrix[:, 0]
        cos_angle = np.dot(approach_dir, vertical_dir)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 将3D抓取点投影到图像平面
        grasp_point_3d = grasp.translation  # 相机坐标系下的3D点
        x_3d, y_3d, z_3d = grasp_point_3d
        
        # 高度约束：确保抓取点不低于物体最低点
        if object_min_z is not None and z_3d < object_min_z:
            continue  # 跳过低于物体最低点的抓取
        
        # 使用针孔相机模型将3D点投影到图像平面
        if z_3d > 0:  # 确保深度值大于0
            u = int((x_3d * fx) / z_3d + cx)
            v = int((y_3d * fy) / z_3d + cy)
            
            # 计算投影点与最近目标中心的距离
            min_distance = float('inf')
            for center in target_centers:
                center_x, center_y = center
                distance = np.sqrt((u - center_x)**2 + (v - center_y)**2)
                if distance < min_distance:
                    min_distance = distance
        else:
            min_distance = float('inf')
        
        grasp_with_info.append((grasp, np.rad2deg(angle), min_distance, z_3d))
        valid_grasp_count += 1
    
    if valid_grasp_count == 0 and len(grasp_group) > 0:
        print("警告：所有抓取点都低于物体最低点，忽略高度约束...")
        # 如果所有抓取都不满足高度约束，则不应用该约束
        grasp_with_info = []  # 重新计算
        for grasp in grasp_group:
            approach_dir = grasp.rotation_matrix[:, 0]
            cos_angle = np.dot(approach_dir, vertical_dir)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # 将3D抓取点投影到图像平面
            grasp_point_3d = grasp.translation  # 相机坐标系下的3D点
            x_3d, y_3d, z_3d = grasp_point_3d
            
            # 使用针孔相机模型将3D点投影到图像平面
            if z_3d > 0:  # 确保深度值大于0
                u = int((x_3d * fx) / z_3d + cx)
                v = int((y_3d * fy) / z_3d + cy)
                
                # 计算投影点与最近目标中心的距离
                min_distance = float('inf')
                for center in target_centers:
                    center_x, center_y = center
                    distance = np.sqrt((u - center_x)**2 + (v - center_y)**2)
                    if distance < min_distance:
                        min_distance = distance
            else:
                min_distance = float('inf')
            
            grasp_with_info.append((grasp, np.rad2deg(angle), min_distance, z_3d))
    
    # 按照夹角和距离综合排序
    # 角度权重0.3，距离权重0.7（优先考虑靠近目标中心）
    angles = [x[1] for x in grasp_with_info]
    distances = [x[2] for x in grasp_with_info]
    
    # 归一化角度（角度越小越好）
    norm_angles = [(90 - angle) / 90 for angle in angles]
    
    # 归一化距离（距离越小越好）
    max_distance = max(distances) if max(distances) > 0 else 1
    # 对于无效距离（无穷大），设为最大距离
    norm_distances = []
    for distance in distances:
        if distance == float('inf'):
            norm_distances.append(0)  # 无穷大距离给予最低评分
        else:
            norm_distances.append(1 - (distance / max_distance))
    
    # 综合评分
    combined_scores = [0.3 * norm_angles[i] + 0.7 * norm_distances[i] for i in range(len(norm_angles))]
    
    # 添加综合评分到列表
    for i in range(len(grasp_with_info)):
        grasp_with_info[i] = (*grasp_with_info[i], combined_scores[i])
    
    # 按综合评分排序
    grasp_with_info.sort(key=lambda x: x[4], reverse=True)  # 按综合评分排序（索引4）
    
    print(f"抓取信息统计：")
    print(f"  总数: {len(grasp_with_info)}")
    if len(grasp_with_info) > 0:
        print(f"  最小角度: {min([x[1] for x in grasp_with_info]):.2f}°")
        print(f"  最大角度: {max([x[1] for x in grasp_with_info]):.2f}°")
        print(f"  平均角度: {np.mean([x[1] for x in grasp_with_info]):.2f}°")
        valid_distances = [x[2] for x in grasp_with_info if x[2] != float('inf')]
        if valid_distances:
            print(f"  最小距离: {min(valid_distances):.2f} pixels")
            print(f"  最大距离: {max(valid_distances):.2f} pixels")
            print(f"  平均距离: {np.mean(valid_distances):.2f} pixels")
        else:
            print("  没有有效的距离数据")
        
        # 显示前10个最佳抓取的信息
        print("\n前10个最佳抓取（综合评分）：")
        for i, (grasp, angle, distance, height, score) in enumerate(grasp_with_info[:10]):
            distance_str = f"{distance:.2f}" if distance != float('inf') else "无效"
            print(f" {i+1}. 角度: {angle:.2f}°, 距离: {distance_str}px, 高度: {height:.4f}m, 综合评分: {score:.4f}, 得分: {grasp.score:.4f}")
    
    # 7. 选择最优抓取
    if len(grasp_with_info) > 0:
        best_grasp, best_angle, best_distance, best_height, best_score = grasp_with_info[0]
        print(f"\n选择综合评分最高的抓取:")
        distance_str = f"{best_distance:.2f}" if best_distance != float('inf') else "无效"
        print(f"   - 角度: {best_angle:.2f}°")
        print(f"   - 距离目标中心: {distance_str}px")
        print(f"   - 抓取高度: {best_height:.4f}m")
        if object_min_z is not None:
            print(f"   - 物体最低点: {object_min_z:.4f}m")
            print(f"   - 高度差: {(best_height - object_min_z):.4f}m")
        print(f"   - 综合评分: {best_score:.4f}")
        print(f"   - 置信度得分: {best_grasp.score:.4f}")
        
        # 获取前1个候选
        top_grasps = [grasp_with_info[0][0]]
    else:
        print("错误：无有效抓取位姿！")
        return None
    
    # 8. 可视化抓取结果
    grippers = [g.to_open3d_geometry() for g in top_grasps]
    print("=== 可视化抓取结果（关闭窗口继续）===")
    o3d.visualization.draw_geometries([cloud_o3d, *grippers], window_name="Grasp Predictions")
    
    # 9. 提取最优抓取位姿并转换坐标
    best_grasp = top_grasps[0]  # 此时top_grasps一定非空，不会报错
    best_trans_cam = best_grasp.translation
    best_rot_mat_cam = best_grasp.rotation_matrix
    best_width = best_grasp.width
    
    # 10. 坐标转换
    best_pose_base = convert_grasp_to_robot_base(
        grasp_translation=best_trans_cam,
        grasp_rotation_mat=best_rot_mat_cam
    )
    
    # 创建格式化字符串列表
    formatted_values = []
    for value in best_pose_base:
        # 保留6位小数，总宽度12字符（包含符号和空格）
        # 正数前留4空格，负数前留3空格，逗号后留1空格
        if value >= 0:
            formatted_value = f"   {value:9.6f}"
        else:
            formatted_value = f"  {value:9.6f}"
        formatted_values.append(formatted_value)
    
    # 组合成带逗号的数组格式
    formatted_pose = "[" + ",".join(formatted_values) + "]"
    
    distance_str = f"{best_distance:.2f}" if best_distance != float('inf') else "无效"
    print("\n=== 抓取位姿结果汇总 ===")
    print("1. 相机坐标系下最优抓取位姿：")
    print(f"   - 平移 (x,y,z): {best_trans_cam.round(6)} (m)")
    print(f"   - 旋转矩阵:\n{best_rot_mat_cam.round(6)}")
    print(f"   - 抓取宽度: {best_width:.6f} (m)")
    print(f"   - 置信度得分: {best_grasp.score:.6f}")
    print(f"   - 与垂直方向夹角: {best_angle:.2f}°")
    print(f"   - 距离目标中心: {distance_str}px")
    print(f"   - 抓取高度: {best_trans_cam[2]:.4f}m")
    if object_min_z is not None:
        print(f"   - 物体最低点高度: {object_min_z:.4f}m")
        print(f"   - 高度差: {(best_trans_cam[2] - object_min_z):.4f}m")
    print("\n2. 机械臂基座坐标系下目标位姿：")
    print(f"   - 位姿 [x,y,z,rx,ry,rz]: {formatted_pose}")
    
    return best_trans_cam, best_rot_mat_cam, best_width, best_pose_base, top_grasps

# 自动版本
def run_grasp_prediction_auto(grasp_net, color_path, depth_path, mask_path):
    """执行抓取位姿预测，并调用坐标转换函数，返回：最优抓取位姿+基座坐标系位姿（自动模式，无用户交互）"""
    print("\n=== 开始抓取位姿预测 ===")
    device = next(grasp_net.parameters()).device
    
    # 1. 处理输入数据（彩色图+深度图+掩码→点云）
    end_points, cloud_o3d = get_and_process_grasp_data(color_path, depth_path, mask_path)
    
    # 获取点云数据用于计算物体最低点
    point_cloud_points = np.asarray(cloud_o3d.points)
    if len(point_cloud_points) > 0:
        # 物体最低点（最小z坐标）
        object_min_z = np.min(point_cloud_points[:, 2])
        print(f"物体最低点高度: {object_min_z:.6f}m")
    else:
        object_min_z = None
    
    # 获取YOLO检测框中心点
    color_img = np.array(Image.open(color_path))
    yolo_model = YOLO(Config.YOLO_MODEL_PATH)
    results = yolo_model(color_img, conf=0.3)
    
    target_centers = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            target_centers.append((center_x, center_y))
    
    # 2. GraspNet前向推理
    with torch.no_grad():
        end_points = grasp_net(end_points)
        grasp_preds = pred_decode(end_points)
    
    # 3. 构建原始抓取组并检查是否为空
    grasp_group = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    if len(grasp_group) == 0:
        print("错误：GraspNet未预测到任何抓取位姿！请检查输入点云或模型权重。")
        return None  # 提前返回，避免后续错误
    
    # 4. 碰撞检测（并检查结果）
    if Config.COLLISION_THRESH > 0:
        grasp_group = collision_detection(grasp_group, np.asarray(cloud_o3d.points))
        if len(grasp_group) == 0:
            print("警告：所有抓取位姿均与背景碰撞！尝试降低 COLLISION_THRESH 参数。")
            # 若碰撞检测后为空，使用原始抓取组重试
            grasp_group = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    
    # 5. NMS去重
    grasp_group.nms()
    
    # 6. 根据夹爪夹角大小、距离目标中心的远近和高度约束排序
    print("=== 根据夹爪夹角大小、距离目标中心和高度约束排序 ===")
    vertical_dir = np.array([0, 0, 1])
    
    # 获取相机内参
    fx = Config.DEPTH_INTR['fx']
    fy = Config.DEPTH_INTR['fy']
    cx = Config.DEPTH_INTR['ppx']
    cy = Config.DEPTH_INTR['ppy']
    
    # 计算每个抓取与垂直方向的夹角，并存储抓取和角度信息
    grasp_with_info = []
    valid_grasp_count = 0
    
    for grasp in grasp_group:
        approach_dir = grasp.rotation_matrix[:, 0]
        cos_angle = np.dot(approach_dir, vertical_dir)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 将3D抓取点投影到图像平面
        grasp_point_3d = grasp.translation  # 相机坐标系下的3D点
        x_3d, y_3d, z_3d = grasp_point_3d
        
        # 高度约束：确保抓取点不低于物体最低点
        if object_min_z is not None and z_3d < object_min_z:
            continue  # 跳过低于物体最低点的抓取
        
        # 使用针孔相机模型将3D点投影到图像平面
        if z_3d > 0:  # 确保深度值大于0
            u = int((x_3d * fx) / z_3d + cx)
            v = int((y_3d * fy) / z_3d + cy)
            
            # 计算投影点与最近目标中心的距离
            min_distance = float('inf')
            for center in target_centers:
                center_x, center_y = center
                distance = np.sqrt((u - center_x)**2 + (v - center_y)**2)
                if distance < min_distance:
                    min_distance = distance
        else:
            min_distance = float('inf')
        
        grasp_with_info.append((grasp, np.rad2deg(angle), min_distance, z_3d))
        valid_grasp_count += 1
    
    if valid_grasp_count == 0 and len(grasp_group) > 0:
        print("警告：所有抓取点都低于物体最低点，忽略高度约束...")
        # 如果所有抓取都不满足高度约束，则不应用该约束
        grasp_with_info = []  # 重新计算
        for grasp in grasp_group:
            approach_dir = grasp.rotation_matrix[:, 0]
            cos_angle = np.dot(approach_dir, vertical_dir)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # 将3D抓取点投影到图像平面
            grasp_point_3d = grasp.translation  # 相机坐标系下的3D点
            x_3d, y_3d, z_3d = grasp_point_3d
            
            # 使用针孔相机模型将3D点投影到图像平面
            if z_3d > 0:  # 确保深度值大于0
                u = int((x_3d * fx) / z_3d + cx)
                v = int((y_3d * fy) / z_3d + cy)
                
                # 计算投影点与最近目标中心的距离
                min_distance = float('inf')
                for center in target_centers:
                    center_x, center_y = center
                    distance = np.sqrt((u - center_x)**2 + (v - center_y)**2)
                    if distance < min_distance:
                        min_distance = distance
            else:
                min_distance = float('inf')
            
            grasp_with_info.append((grasp, np.rad2deg(angle), min_distance, z_3d))
    
    # 按照夹角和距离综合排序
    # 角度权重0.3，距离权重0.7（优先考虑靠近目标中心）
    angles = [x[1] for x in grasp_with_info]
    distances = [x[2] for x in grasp_with_info]
    
    # 归一化角度（角度越小越好）
    norm_angles = [(90 - angle) / 90 for angle in angles]
    
    # 归一化距离（距离越小越好）
    max_distance = max(distances) if max(distances) > 0 else 1
    # 对于无效距离（无穷大），设为最大距离
    norm_distances = []
    for distance in distances:
        if distance == float('inf'):
            norm_distances.append(0)  # 无穷大距离给予最低评分
        else:
            norm_distances.append(1 - (distance / max_distance))
    
    # 综合评分
    combined_scores = [0.3 * norm_angles[i] + 0.7 * norm_distances[i] for i in range(len(norm_angles))]
    
    # 添加综合评分到列表
    for i in range(len(grasp_with_info)):
        grasp_with_info[i] = (*grasp_with_info[i], combined_scores[i])
    
    # 按综合评分排序
    grasp_with_info.sort(key=lambda x: x[4], reverse=True)  # 按综合评分排序（索引4）
    
    print(f"抓取信息统计：")
    print(f"  总数: {len(grasp_with_info)}")
    if len(grasp_with_info) > 0:
        print(f"  最小角度: {min([x[1] for x in grasp_with_info]):.2f}°")
        print(f"  最大角度: {max([x[1] for x in grasp_with_info]):.2f}°")
        print(f"  平均角度: {np.mean([x[1] for x in grasp_with_info]):.2f}°")
        valid_distances = [x[2] for x in grasp_with_info if x[2] != float('inf')]
        if valid_distances:
            print(f"  最小距离: {min(valid_distances):.2f} pixels")
            print(f"  最大距离: {max(valid_distances):.2f} pixels")
            print(f"  平均距离: {np.mean(valid_distances):.2f} pixels")
        else:
            print("  没有有效的距离数据")
        
        # 显示前10个最佳抓取的信息
        print("\n前10个最佳抓取（综合评分）：")
        for i, (grasp, angle, distance, height, score) in enumerate(grasp_with_info[:10]):
            distance_str = f"{distance:.2f}" if distance != float('inf') else "无效"
            print(f" {i+1}. 角度: {angle:.2f}°, 距离: {distance_str}px, 高度: {height:.4f}m, 综合评分: {score:.4f}, 得分: {grasp.score:.4f}")
    
    # 7. 选择最优抓取
    if len(grasp_with_info) > 0:
        best_grasp, best_angle, best_distance, best_height, best_score = grasp_with_info[0]
        print(f"\n选择综合评分最高的抓取:")
        distance_str = f"{best_distance:.2f}" if best_distance != float('inf') else "无效"
        print(f"   - 角度: {best_angle:.2f}°")
        print(f"   - 距离目标中心: {distance_str}px")
        print(f"   - 抓取高度: {best_height:.4f}m")
        if object_min_z is not None:
            print(f"   - 物体最低点: {object_min_z:.4f}m")
            print(f"   - 高度差: {(best_height - object_min_z):.4f}m")
        print(f"   - 综合评分: {best_score:.4f}")
        print(f"   - 置信度得分: {best_grasp.score:.4f}")
        
        # 获取前1个候选
        top_grasps = [grasp_with_info[0][0]]
    else:
        print("错误：无有效抓取位姿！")
        return None
    
    # 8. 跳过可视化抓取结果（自动模式下不显示）
    print("=== 跳过可视化抓取结果（自动模式）===")
    
    # 9. 提取最优抓取位姿并转换坐标
    best_grasp = top_grasps[0]  # 此时top_grasps一定非空，不会报错
    best_trans_cam = best_grasp.translation
    best_rot_mat_cam = best_grasp.rotation_matrix
    best_width = best_grasp.width
    
    # 10. 坐标转换
    best_pose_base = convert_grasp_to_robot_base(
        grasp_translation=best_trans_cam,
        grasp_rotation_mat=best_rot_mat_cam
    )
    
    # 创建格式化字符串列表
    formatted_values = []
    for value in best_pose_base:
        # 保留6位小数，总宽度12字符（包含符号和空格）
        # 正数前留4空格，负数前留3空格，逗号后留1空格
        if value >= 0:
            formatted_value = f"   {value:9.6f}"
        else:
            formatted_value = f"  {value:9.6f}"
        formatted_values.append(formatted_value)
    
    # 组合成带逗号的数组格式
    formatted_pose = "[" + ",".join(formatted_values) + "]"
    
    distance_str = f"{best_distance:.2f}" if best_distance != float('inf') else "无效"
    print("\n=== 抓取位姿结果汇总 ===")
    print("1. 相机坐标系下最优抓取位姿：")
    print(f"   - 平移 (x,y,z): {best_trans_cam.round(6)} (m)")
    print(f"   - 旋转矩阵:\n{best_rot_mat_cam.round(6)}")
    print(f"   - 抓取宽度: {best_width:.6f} (m)")
    print(f"   - 置信度得分: {best_grasp.score:.6f}")
    print(f"   - 与垂直方向夹角: {best_angle:.2f}°")
    print(f"   - 距离目标中心: {distance_str}px")
    print(f"   - 抓取高度: {best_trans_cam[2]:.4f}m")
    if object_min_z is not None:
        print(f"   - 物体最低点高度: {object_min_z:.4f}m")
        print(f"   - 高度差: {(best_trans_cam[2] - object_min_z):.4f}m")
    print("\n2. 机械臂基座坐标系下目标位姿：")
    print(f"   - 位姿 [x,y,z,rx,ry,rz]: {formatted_pose}")
    
    return best_trans_cam, best_rot_mat_cam, best_width, best_pose_base, top_grasps

# -------------------------- 第七步：GraspNet辅助函数（原逻辑保留） --------------------------
def get_and_process_grasp_data(color_path, depth_path, mask_path):
    """处理抓取预测的输入数据，返回模型输入和Open3D点云"""
    # 加载图像
    color = np.array(Image.open(color_path), dtype=np.float32) / 255.0  # 归一化到[0,1]
    depth = np.array(Image.open(depth_path), dtype=np.uint16)          # 保留16位深度
    mask = np.array(Image.open(mask_path), dtype=np.uint8)             # 掩码（0/255）
    
    # 相机内参配置
    camera = CameraInfo(
        width=Config.CAMERA_RES[0],
        height=Config.CAMERA_RES[1],
        fx=Config.DEPTH_INTR['fx'],
        fy=Config.DEPTH_INTR['fy'],
        cx=Config.DEPTH_INTR['ppx'],
        cy=Config.DEPTH_INTR['ppy'],
        scale=Config.DEPTH_FACTOR
    )
    
    # 生成点云并应用掩码
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    mask_resized = cv2.resize(mask, (depth.shape[1], depth.shape[0]))  # 匹配深度图尺寸
    mask_bool = (mask_resized > 0)
    cloud_masked = cloud[mask_bool]
    color_masked = color[mask_bool]
    
    # 点云采样（确保点数符合要求）
    if len(cloud_masked) >= Config.NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), Config.NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), Config.NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2])
    cloud_sampled = cloud_masked[idxs]
    
    # 转换为Open3D点云（可视化用）
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    
    # 转换为模型输入（Tensor）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': cloud_tensor}
    
    return end_points, cloud_o3d


def collision_detection(grasp_group, cloud_points):
    """抓取位姿碰撞检测，过滤碰撞位姿"""
    mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=Config.VOXEL_SIZE)
    collision_mask = mfcdetector.detect(
        grasp_group, 
        approach_dist=0.05, 
        collision_thresh=Config.COLLISION_THRESH
    )
    return grasp_group[~collision_mask]


# -------------------------- 第八步：主程序（端到端流程控制，原逻辑保留） --------------------------
if __name__ == "__main__":
    # 安装依赖提醒（若未安装scipy）
    try:
        from scipy.spatial.transform import Rotation as R
    except ImportError:
        print("请先安装scipy库：pip install scipy")
        sys.exit(1)
    
    # 1. 加载所有模型
    yolo_model, sam_predictor, grasp_net, device = load_all_models()
    
    # 2. 初始化RealSense相机/ROS Bag
    pipeline = rs.pipeline()
    config = rs.config()
    
    if Config.USE_ROS_BAG:
        if not os.path.exists(Config.BAG_PATH):
            raise FileNotFoundError(f"ROS Bag文件不存在：{Config.BAG_PATH}")
        config.enable_device_from_file(Config.BAG_PATH)
        print(f"已加载ROS Bag：{Config.BAG_PATH}")
    else:
        # 实时相机流配置（彩色+深度+红外）
        config.enable_stream(rs.stream.color, Config.CAMERA_RES[0], Config.CAMERA_RES[1], rs.format.bgr8, Config.CAMERA_FPS)
        config.enable_stream(rs.stream.depth, Config.CAMERA_RES[0], Config.CAMERA_RES[1], rs.format.z16, Config.CAMERA_FPS)
        config.enable_stream(rs.stream.infrared, 1, Config.CAMERA_RES[0], Config.CAMERA_RES[1], rs.format.y8, Config.CAMERA_FPS)
        config.enable_stream(rs.stream.infrared, 2, Config.CAMERA_RES[0], Config.CAMERA_RES[1], rs.format.y8, Config.CAMERA_FPS)
        print("已启动RealSense实时相机")
    
    # 3. 初始化图像对齐器
    align_to = rs.stream.color if Config.ALIGN_WAY else rs.stream.depth
    aligner = rs.align(align_to)
    
    # 4. 启动相机管道
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例系数：{depth_scale:.6f} 米/像素\n")
    
    # 5. 主循环
    save_count = 0
    print("=== 操作说明 ===")
    print("按 's' 键：保存图像→生成掩码→抓取预测→坐标转换")
    print("按 'q' 或 ESC 键：退出程序\n")
    
    try:
        while True:
            # 获取相机帧
            frames = pipeline.wait_for_frames()
            
            # 处理对齐帧并显示
            color_aligned, depth_aligned, depth_colormap = process_aligned_frames(
                frames, aligner, Config.USE_ROS_BAG
            )
            # 处理未对齐帧（红外显示）
            process_unaligned_frames(frames, Config.USE_ROS_BAG, Config.ALIGN_WAY)
            
            # 显示对齐后的彩色图+深度图
            align_visual = np.hstack((color_aligned, depth_colormap))
            cv2.imshow('Aligned Images (Color + Depth)', align_visual)
            
            # 按键交互
            key = cv2.waitKey(1)
            if key == ord('s'):
                # 保存对齐图像
                color_save_path = os.path.join(Config.COLOR_SAVE_DIR, f"aligned_color_{save_count:04d}.png")
                depth_save_path = os.path.join(Config.DEPTH_SAVE_DIR, f"aligned_depth_{save_count:04d}.png")
                
                # 保存彩色图（RGB→BGR，适配OpenCV保存）
                cv2.imwrite(color_save_path, cv2.cvtColor(color_aligned, cv2.COLOR_RGB2BGR))
                # 保存深度图（16位格式）
                cv2.imwrite(depth_save_path, depth_aligned.astype(np.uint16))
                print(f"\n=== 已保存第 {save_count+1} 组图像 ===")
                print(f"彩色图：{color_save_path}")
                print(f"深度图：{depth_save_path}")
                
                # 生成掩码
                sam_mask_path, yolo_mask_path, cls_n = generate_masks(
                    color_aligned, color_save_path, yolo_model, sam_predictor, device
                )
                
                # 选择掩码
                mask_path = yolo_mask_path if Config.MASK_CHOICE == 1 else sam_mask_path
                print(f"使用掩码类型：{'YOLO扩展掩码' if Config.MASK_CHOICE == 1 else 'SAM分割掩码'}")
                
                # 执行抓取预测+坐标转换（核心整合点）
                run_grasp_prediction(grasp_net, color_save_path, depth_save_path, mask_path)
                
                save_count += 1
                
            elif key in [27, ord('q')]:
                print("\n=== 程序退出 ===")
                break
                
    finally:
        # 释放资源
        pipeline.stop()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()
        print("资源已释放")
