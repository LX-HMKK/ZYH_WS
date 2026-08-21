#coding=utf-8
"""运行时配置与路径解析。"""
import os
import sys
import numpy as np
import yaml

from arm_utils import get_workspace_root


# 配置GraspNet依赖路径（models/ 与 utils/ 直接加入 sys.path 以兼容原 GraspNet 导入方式）
WORKSPACE_ROOT = get_workspace_root()
sys.path.append(os.path.join(WORKSPACE_ROOT, 'tools', 'graspnet_baseline', 'models'))
sys.path.append(os.path.join(WORKSPACE_ROOT, 'tools', 'graspnet_baseline', 'utils'))


def resolve_path(path: str) -> str:
    """将路径中的 {ROBOARM_WS} 占位符替换为实际根目录。"""
    return path.replace("{ROBOARM_WS}", get_workspace_root())


class Config:
    """从 grasp_config.yaml 加载的运行时配置。"""

    _loaded = False

    @classmethod
    def load(cls, config_path: str | None = None):
        """加载 YAML 配置。若未指定路径，默认读取仓库中的 grasp_config.yaml。"""
        if config_path is None:
            config_path = f"{get_workspace_root()}/src/vision_grasp/config/grasp_config.yaml"

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 相机与对齐配置
        cls.USE_ROS_BAG = cfg.get("use_ros_bag", 0)
        cls.ALIGN_WAY = cfg.get("align_way", 1)
        cls.BAG_PATH = cfg.get("bag_path", "666.bag")
        cls.CAMERA_RES = tuple(cfg.get("camera_res", [1280, 720]))
        cls.CAMERA_FPS = cfg.get("camera_fps", 30)

        # 模型路径
        cls.YOLO_MODEL_PATH = resolve_path(cfg.get("yolo_model_path", ""))
        cls.SAM_CHECKPOINT = resolve_path(cfg.get("sam_checkpoint", ""))
        cls.SAM_MODEL_TYPE = cfg.get("sam_model_type", "vit_b")
        cls.GRASP_CHECKPOINT = resolve_path(cfg.get("grasp_checkpoint", ""))

        # 输出目录
        cls.OUTPUT_ROOT = cfg.get("output_root", "aligned_images")
        cls.COLOR_SAVE_DIR = os.path.join(cls.OUTPUT_ROOT, "color")
        cls.DEPTH_SAVE_DIR = os.path.join(cls.OUTPUT_ROOT, "depth")
        cls.MASK_SAVE_DIR = os.path.join(cls.OUTPUT_ROOT, "masks")

        # GraspNet 抓取参数
        cls.NUM_POINT = cfg.get("num_point", 10000)
        cls.NUM_VIEW = cfg.get("num_view", 300)
        cls.COLLISION_THRESH = cfg.get("collision_thresh", 0.001)
        cls.VOXEL_SIZE = cfg.get("voxel_size", 0.001)
        cls.DEPTH_FACTOR = cfg.get("depth_factor", 1000.0)

        intr = cfg.get("depth_intr", {})
        cls.DEPTH_INTR = {
            "ppx": intr.get("ppx", 644.136),
            "ppy": intr.get("ppy", 354.556),
            "fx": intr.get("fx", 902.806),
            "fy": intr.get("fy", 900.776),
        }
        cls.MASK_CHOICE = cfg.get("mask_choice", 0)

        # 机械臂与手眼标定参数
        cls.CURRENT_EE_POSE = cfg.get("current_ee_pose", [-0.115955, -0.320591, -0.428274, -0.00347, -0.0538, 0.0212])
        cls.GRIPPER_LENGTH = cfg.get("gripper_length", -0.14)
        cls.HANDEYE_ROT = np.array(cfg.get("handeye_rot", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), dtype=float)
        cls.HANDEYE_TRANS = np.array(cfg.get("handeye_trans", [0.0, 0.0, 0.0]), dtype=float)

        cls._loaded = True


# 默认加载配置
Config.load()

# 初始化输出目录
os.makedirs(Config.COLOR_SAVE_DIR, exist_ok=True)
os.makedirs(Config.DEPTH_SAVE_DIR, exist_ok=True)
os.makedirs(Config.MASK_SAVE_DIR, exist_ok=True)
