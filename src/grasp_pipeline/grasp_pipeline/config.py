#coding=utf-8
"""运行时配置与路径解析。"""
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

from arm_utils import get_workspace_root


def resolve_path(path: str) -> str:
    """将路径中的 {ROBOARM_WS} 占位符替换为实际根目录。"""
    return path.replace("{ROBOARM_WS}", get_workspace_root())


@dataclass(eq=False)
class Config:
    """从 grasp_config.yaml 加载的运行时配置（实例化配置对象）。"""

    # 相机与对齐配置
    USE_ROS_BAG: int = 0
    ALIGN_WAY: int = 1
    BAG_PATH: str = "666.bag"
    CAMERA_RES: tuple = field(default_factory=lambda: (1280, 720))
    CAMERA_FPS: int = 30

    # 模型路径
    YOLO_MODEL_PATH: str = ""
    SAM_CHECKPOINT: str = ""
    SAM_MODEL_TYPE: str = "vit_b"
    GRASP_CHECKPOINT: str = ""

    # 输出目录
    OUTPUT_ROOT: str = "aligned_images"
    COLOR_SAVE_DIR: str = "aligned_images/color"
    DEPTH_SAVE_DIR: str = "aligned_images/depth"
    MASK_SAVE_DIR: str = "aligned_images/masks"

    # GraspNet 抓取参数
    NUM_POINT: int = 10000
    NUM_VIEW: int = 300
    COLLISION_THRESH: float = 0.001
    VOXEL_SIZE: float = 0.001
    DEPTH_FACTOR: float = 1000.0

    DEPTH_INTR: dict = field(default_factory=lambda: {
        "ppx": 644.136,
        "ppy": 354.556,
        "fx": 902.806,
        "fy": 900.776,
    })
    MASK_CHOICE: int = 0

    # 机械臂与手眼标定参数
    CURRENT_EE_POSE: list = field(
        default_factory=lambda: [-0.115955, -0.320591, -0.428274, -0.00347, -0.0538, 0.0212]
    )
    GRIPPER_LENGTH: float = -0.14
    HANDEYE_ROT: np.ndarray = field(
        default_factory=lambda: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    )
    HANDEYE_TRANS: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )

    @classmethod
    def load_yaml(cls, path: str) -> dict:
        """加载 YAML 文件并返回字典。"""
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def load(cls, config_path: str | None = None) -> "Config":
        """加载 YAML 配置。若未指定路径，默认读取仓库中的 grasp_config.yaml。"""
        if config_path is None:
            config_path = f"{get_workspace_root()}/src/grasp_pipeline/config/grasp_config.yaml"

        cfg = cls.load_yaml(config_path)

        output_root = cfg.get("output_root", "aligned_images")
        color_save_dir = os.path.join(output_root, "color")
        depth_save_dir = os.path.join(output_root, "depth")
        mask_save_dir = os.path.join(output_root, "masks")

        os.makedirs(color_save_dir, exist_ok=True)
        os.makedirs(depth_save_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)

        intr = cfg.get("depth_intr", {})
        return cls(
            USE_ROS_BAG=cfg.get("use_ros_bag", 0),
            ALIGN_WAY=cfg.get("align_way", 1),
            BAG_PATH=cfg.get("bag_path", "666.bag"),
            CAMERA_RES=tuple(cfg.get("camera_res", [1280, 720])),
            CAMERA_FPS=cfg.get("camera_fps", 30),
            YOLO_MODEL_PATH=resolve_path(cfg.get("yolo_model_path", "")),
            SAM_CHECKPOINT=resolve_path(cfg.get("sam_checkpoint", "")),
            SAM_MODEL_TYPE=cfg.get("sam_model_type", "vit_b"),
            GRASP_CHECKPOINT=resolve_path(cfg.get("grasp_checkpoint", "")),
            OUTPUT_ROOT=output_root,
            COLOR_SAVE_DIR=color_save_dir,
            DEPTH_SAVE_DIR=depth_save_dir,
            MASK_SAVE_DIR=mask_save_dir,
            NUM_POINT=cfg.get("num_point", 10000),
            NUM_VIEW=cfg.get("num_view", 300),
            COLLISION_THRESH=cfg.get("collision_thresh", 0.001),
            VOXEL_SIZE=cfg.get("voxel_size", 0.001),
            DEPTH_FACTOR=cfg.get("depth_factor", 1000.0),
            DEPTH_INTR={
                "ppx": intr.get("ppx", 644.136),
                "ppy": intr.get("ppy", 354.556),
                "fx": intr.get("fx", 902.806),
                "fy": intr.get("fy", 900.776),
            },
            MASK_CHOICE=cfg.get("mask_choice", 0),
            CURRENT_EE_POSE=cfg.get(
                "current_ee_pose",
                [-0.115955, -0.320591, -0.428274, -0.00347, -0.0538, 0.0212],
            ),
            GRIPPER_LENGTH=cfg.get("gripper_length", -0.14),
            HANDEYE_ROT=np.array(
                cfg.get("handeye_rot", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), dtype=float
            ),
            HANDEYE_TRANS=np.array(
                cfg.get("handeye_trans", [0.0, 0.0, 0.0]), dtype=float
            ),
        )
