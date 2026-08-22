#coding=utf-8
"""YOLO、SAM、GraspNet 模型加载。"""
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

import grasp_pipeline._graspnet_baseline_path as _  # noqa: F401  # 加载 GraspNet 路径 shim
from graspnet import GraspNet
from ..config import Config


class ModelManager:
    """统一加载视觉检测与抓取预测所需的模型。"""

    def __init__(self, config: Config | None = None):
        if config is None:
            raise ValueError("ModelManager 必须注入 Config 实例")
        self.config = config

    def load_all(self) -> tuple:
        """
        加载 YOLO、SAM、GraspNet 模型。

        Returns:
            (yolo_model, sam_predictor, grasp_net, device)
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"=== 模型加载（设备：{device}）===")

        # 1. 加载 YOLO 检测模型
        print("1. 加载YOLO模型...")
        yolo_model = YOLO(self.config.YOLO_MODEL_PATH)
        yolo_model.to(device)

        # 2. 加载 SAM 分割模型
        print("2. 加载SAM模型...")
        sam = sam_model_registry[self.config.SAM_MODEL_TYPE](checkpoint=self.config.SAM_CHECKPOINT)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)

        # 3. 加载 GraspNet 抓取模型
        print("3. 加载GraspNet模型...")
        grasp_net = GraspNet(
            input_feature_dim=0,
            num_view=self.config.NUM_VIEW,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False
        )
        grasp_net.to(device)
        checkpoint = torch.load(self.config.GRASP_CHECKPOINT, map_location=device)
        grasp_net.load_state_dict(checkpoint['model_state_dict'])
        grasp_net.eval()

        print("=== 所有模型加载完成 ===\n")
        return yolo_model, sam_predictor, grasp_net, device


# 保持与原函数签名兼容的模块级函数
def load_all_models(config: Config | None = None):
    if config is None:
        config = Config.load()
    return ModelManager(config).load_all()
