#coding=utf-8
"""相机坐标系到机械臂基坐标系的位姿转换。"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..config import Config


class CoordinateTransformer:
    """将 GraspNet 输出的相机坐标系抓取位姿转换到机械臂基坐标系。"""

    def __init__(self, config: Config):
        if config is None:
            raise ValueError("CoordinateTransformer 必须注入 Config 实例")
        self.config = config

    def grasp_to_base(
        self,
        grasp_translation: np.ndarray,
        grasp_rotation_mat: np.ndarray,
        current_ee_pose=None,
        handeye_rot=None,
        handeye_trans=None,
        gripper_length=None,
    ) -> np.ndarray:
        """
        将相机坐标系下的抓取位姿转换到机械臂基坐标系。

        Args:
            grasp_translation: 相机坐标系下的抓取平移 (m)
            grasp_rotation_mat: 相机坐标系下的抓取旋转矩阵 (3x3)
            current_ee_pose: 机械臂当前末端位姿，默认使用 config.CURRENT_EE_POSE
            handeye_rot: 手眼标定旋转矩阵，默认使用 config.HANDEYE_ROT
            handeye_trans: 手眼标定平移向量，默认使用 config.HANDEYE_TRANS
            gripper_length: 夹爪长度补偿，默认使用 config.GRIPPER_LENGTH

        Returns:
            [x, y, z, rx, ry, rz] 基坐标系下的位姿
        """
        if current_ee_pose is None:
            current_ee_pose = self.config.CURRENT_EE_POSE
        if handeye_rot is None:
            handeye_rot = self.config.HANDEYE_ROT
        if handeye_trans is None:
            handeye_trans = self.config.HANDEYE_TRANS
        if gripper_length is None:
            gripper_length = self.config.GRIPPER_LENGTH

        # 坐标系对齐矩阵
        r_adjust = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ], dtype=np.float32)

        t_align = np.eye(4, dtype=float)
        t_align[:3, :3] = r_adjust

        # 抓取位姿到相机的变换矩阵（应用坐标系对齐）
        t_grasp2cam = np.eye(4)
        t_grasp2cam[:3, :3] = grasp_rotation_mat
        t_grasp2cam[:3, 3] = grasp_translation
        t_grasp2cam = t_grasp2cam @ t_align

        # 手眼标定矩阵（相机→末端）
        t_cam2end = np.eye(4)
        t_cam2end[:3, :3] = handeye_rot
        t_cam2end[:3, 3] = handeye_trans

        # 末端到基座变换
        x, y, z, rx, ry, rz = current_ee_pose
        r_end2base = R.from_euler('XYZ', [rx, ry, rz]).as_matrix()
        t_end2base = np.eye(4)
        t_end2base[:3, :3] = r_end2base
        t_end2base[:3, 3] = [x, y, z]

        # 基座到抓取的完整变换链
        t_base2grasp = t_end2base @ t_cam2end @ t_grasp2cam

        # 如果指定了夹爪长度，则考虑夹爪长度对末端位姿的影响
        if gripper_length:
            t_base2end_final = t_base2grasp.copy()
            t_base2end_final[2, 3] += gripper_length
            final_trans = t_base2end_final[:3, 3]
            final_rot = R.from_matrix(t_base2end_final[:3, :3])
        else:
            final_trans = t_base2grasp[:3, 3]
            final_rot = R.from_matrix(t_base2grasp[:3, :3])

        base_rx, base_ry, base_rz = final_rot.as_euler('XYZ')
        return np.concatenate([final_trans, [base_rx, base_ry, base_rz]])


# 保持与原函数签名兼容的模块级函数
def convert_grasp_to_robot_base(
    grasp_translation,
    grasp_rotation_mat,
    current_ee_pose=None,
    handeye_rot=None,
    handeye_trans=None,
    gripper_length=None,
    config: Config | None = None,
):
    if config is None:
        config = Config.load()
    return CoordinateTransformer(config).grasp_to_base(
        grasp_translation,
        grasp_rotation_mat,
        current_ee_pose,
        handeye_rot,
        handeye_trans,
        gripper_length,
    )
