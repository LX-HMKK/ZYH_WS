#coding=utf-8
"""抓取预测前的点云数据预处理与碰撞检测。"""
import numpy as np
import cv2
import torch
from PIL import Image

from data_utils import CameraInfo, create_point_cloud_from_depth_image
from collision_detector import ModelFreeCollisionDetector
from ..config import Config


class DataProcessor:
    """将彩色/深度/掩码图像转换为 GraspNet 模型输入，并提供碰撞检测。"""

    def __init__(self, config: Config | None = None):
        self.config = config or Config

    def get_and_process_data(self, color_path: str, depth_path: str, mask_path: str):
        """
        处理抓取预测的输入数据，返回模型输入和 Open3D 点云。

        Returns:
            (end_points, cloud_o3d)
        """
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path), dtype=np.uint16)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        camera = CameraInfo(
            width=self.config.CAMERA_RES[0],
            height=self.config.CAMERA_RES[1],
            fx=self.config.DEPTH_INTR['fx'],
            fy=self.config.DEPTH_INTR['fy'],
            cx=self.config.DEPTH_INTR['ppx'],
            cy=self.config.DEPTH_INTR['ppy'],
            scale=self.config.DEPTH_FACTOR
        )

        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        mask_resized = cv2.resize(mask, (depth.shape[1], depth.shape[0]))
        mask_bool = (mask_resized > 0)
        cloud_masked = cloud[mask_bool]
        color_masked = color[mask_bool]

        if len(cloud_masked) >= self.config.NUM_POINT:
            idxs = np.random.choice(len(cloud_masked), self.config.NUM_POINT, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(
                len(cloud_masked),
                self.config.NUM_POINT - len(cloud_masked),
                replace=True
            )
            idxs = np.concatenate([idxs1, idxs2])
        cloud_sampled = cloud_masked[idxs]

        import open3d as o3d
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
        end_points = {'point_clouds': cloud_tensor}

        return end_points, cloud_o3d

    def collision_detection(self, grasp_group, cloud_points):
        """抓取位姿碰撞检测，过滤碰撞位姿。"""
        mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=self.config.VOXEL_SIZE)
        collision_mask = mfcdetector.detect(
            grasp_group,
            approach_dist=0.05,
            collision_thresh=self.config.COLLISION_THRESH
        )
        return grasp_group[~collision_mask]


# 保持与原函数签名兼容的模块级函数
get_and_process_grasp_data = DataProcessor().get_and_process_data
collision_detection = DataProcessor().collision_detection
