#coding=utf-8
"""GraspNet 抓取位姿预测。"""
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from graspnetAPI import GraspGroup
from graspnet import pred_decode
import open3d as o3d

import grasp_pipeline._graspnet_baseline_path as _  # noqa: F401  # 加载 GraspNet 路径 shim
from ..config import Config
from .data_processor import DataProcessor
from ..transforms.coordinate_transformer import CoordinateTransformer


class GraspPredictor:
    """基于 GraspNet 推理、评分与坐标转换，输出最优抓取位姿。"""

    def __init__(self, grasp_net, config: Config | None = None, yolo_model=None):
        if config is None:
            config = Config.load()
        self.grasp_net = grasp_net
        self.config = config
        self.yolo_model = yolo_model
        self.data_processor = DataProcessor(self.config)
        self.transformer = CoordinateTransformer(self.config)

    def predict(self, color_path: str, depth_path: str, mask_path: str, visualize: bool = False):
        """
        执行抓取位姿预测。

        Args:
            color_path: 彩色图路径
            depth_path: 深度图路径
            mask_path: 掩码路径
            visualize: 是否可视化抓取结果（交互模式）

        Returns:
            (best_trans_cam, best_rot_mat_cam, best_width, best_pose_base, top_grasps)
        """
        if visualize:
            print("\n=== 开始抓取位姿预测 ===")

        device = next(self.grasp_net.parameters()).device

        # 1. 处理输入数据
        end_points, cloud_o3d = self.data_processor.get_and_process_data(color_path, depth_path, mask_path)

        # 物体最低点
        point_cloud_points = np.asarray(cloud_o3d.points)
        if len(point_cloud_points) > 0:
            object_min_z = np.min(point_cloud_points[:, 2])
            print(f"物体最低点高度: {object_min_z:.6f}m")
        else:
            object_min_z = None

        # 2. 获取 YOLO 检测框中心点
        target_centers = self._get_target_centers(color_path)

        # 3. GraspNet 前向推理
        with torch.no_grad():
            end_points = self.grasp_net(end_points)
            grasp_preds = pred_decode(end_points)

        # 4. 构建原始抓取组
        grasp_group = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        if len(grasp_group) == 0:
            print("错误：GraspNet未预测到任何抓取位姿！请检查输入点云或模型权重。")
            return None

        # 5. 碰撞检测
        if self.config.COLLISION_THRESH > 0:
            grasp_group = self.data_processor.collision_detection(grasp_group, np.asarray(cloud_o3d.points))
            if len(grasp_group) == 0:
                print("警告：所有抓取位姿均与背景碰撞！尝试降低 COLLISION_THRESH 参数。")
                grasp_group = GraspGroup(grasp_preds[0].detach().cpu().numpy())

        # 6. NMS 去重
        grasp_group.nms()

        # 7. 评分与排序
        grasp_with_info = self._score_grasps(grasp_group, target_centers, object_min_z)

        if len(grasp_with_info) == 0:
            print("错误：无有效抓取位姿！")
            return None

        best_grasp, best_angle, best_distance, best_height, best_score = grasp_with_info[0]
        top_grasps = [grasp_with_info[0][0]]

        self._print_best_grasp_info(best_grasp, best_angle, best_distance, best_height, best_score, object_min_z)

        # 8. 可视化（交互模式）
        if visualize:
            grippers = [g.to_open3d_geometry() for g in top_grasps]
            print("=== 可视化抓取结果（关闭窗口继续）===")
            o3d.visualization.draw_geometries([cloud_o3d, *grippers], window_name="Grasp Predictions")
        else:
            print("=== 跳过可视化抓取结果（自动模式）===")

        # 9. 提取最优抓取位姿并转换坐标
        best_grasp = top_grasps[0]
        best_trans_cam = best_grasp.translation
        best_rot_mat_cam = best_grasp.rotation_matrix
        best_width = best_grasp.width

        best_pose_base = self.transformer.grasp_to_base(
            grasp_translation=best_trans_cam,
            grasp_rotation_mat=best_rot_mat_cam
        )

        self._print_result_summary(best_trans_cam, best_rot_mat_cam, best_width, best_pose_base,
                                   best_angle, best_distance, object_min_z)

        return best_trans_cam, best_rot_mat_cam, best_width, best_pose_base, top_grasps

    def predict_auto(self, color_path: str, depth_path: str, mask_path: str):
        """自动模式预测（不显示窗口）。"""
        return self.predict(color_path, depth_path, mask_path, visualize=False)

    def _get_target_centers(self, color_path: str) -> list:
        """通过 YOLO 获取目标中心点列表。"""
        color_img = np.array(Image.open(color_path))
        if self.yolo_model is None:
            self.yolo_model = YOLO(self.config.YOLO_MODEL_PATH)
        results = self.yolo_model(color_img, conf=0.3)

        target_centers = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                target_centers.append((center_x, center_y))
        return target_centers

    def _score_grasps(self, grasp_group, target_centers, object_min_z):
        """根据夹爪夹角、距离目标中心、高度约束对抓取位姿评分排序。"""
        vertical_dir = np.array([0, 0, 1])
        fx = self.config.DEPTH_INTR['fx']
        fy = self.config.DEPTH_INTR['fy']
        cx = self.config.DEPTH_INTR['ppx']
        cy = self.config.DEPTH_INTR['ppy']

        grasp_with_info = []

        for grasp in grasp_group:
            approach_dir = grasp.rotation_matrix[:, 0]
            cos_angle = np.clip(np.dot(approach_dir, vertical_dir), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            grasp_point_3d = grasp.translation
            x_3d, y_3d, z_3d = grasp_point_3d

            if object_min_z is not None and z_3d < object_min_z:
                continue

            min_distance = self._project_distance_to_targets(x_3d, y_3d, z_3d, target_centers, fx, fy, cx, cy)
            grasp_with_info.append((grasp, np.rad2deg(angle), min_distance, z_3d))

        # 若全部被高度约束过滤，则忽略高度约束重算
        if len(grasp_with_info) == 0 and len(grasp_group) > 0:
            print("警告：所有抓取点都低于物体最低点，忽略高度约束...")
            for grasp in grasp_group:
                approach_dir = grasp.rotation_matrix[:, 0]
                cos_angle = np.clip(np.dot(approach_dir, vertical_dir), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                x_3d, y_3d, z_3d = grasp.translation
                min_distance = self._project_distance_to_targets(x_3d, y_3d, z_3d, target_centers, fx, fy, cx, cy)
                grasp_with_info.append((grasp, np.rad2deg(angle), min_distance, z_3d))

        if len(grasp_with_info) == 0:
            return []

        angles = [x[1] for x in grasp_with_info]
        distances = [x[2] for x in grasp_with_info]

        norm_angles = [(90 - angle) / 90 for angle in angles]
        max_distance = max(distances) if max(distances) > 0 else 1
        norm_distances = [
            0 if d == float('inf') else 1 - (d / max_distance)
            for d in distances
        ]

        combined_scores = [0.3 * norm_angles[i] + 0.7 * norm_distances[i] for i in range(len(norm_angles))]
        grasp_with_info = [(*info, combined_scores[i]) for i, info in enumerate(grasp_with_info)]
        grasp_with_info.sort(key=lambda x: x[4], reverse=True)

        self._print_score_stats(grasp_with_info)
        return grasp_with_info

    @staticmethod
    def _project_distance_to_targets(x_3d, y_3d, z_3d, target_centers, fx, fy, cx, cy) -> float:
        """将 3D 抓取点投影到图像平面并计算到最近目标中心的距离。"""
        if z_3d <= 0:
            return float('inf')
        u = int((x_3d * fx) / z_3d + cx)
        v = int((y_3d * fy) / z_3d + cy)
        return min(
            np.sqrt((u - center_x) ** 2 + (v - center_y) ** 2)
            for center_x, center_y in target_centers
        ) if target_centers else float('inf')

    def _print_score_stats(self, grasp_with_info):
        """打印评分统计信息。"""
        print("=== 根据夹爪夹角大小、距离目标中心和高度约束排序 ===")
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

            print("\n前10个最佳抓取（综合评分）：")
            for i, (grasp, angle, distance, height, score) in enumerate(grasp_with_info[:10]):
                distance_str = f"{distance:.2f}" if distance != float('inf') else "无效"
                print(f" {i+1}. 角度: {angle:.2f}°, 距离: {distance_str}px, 高度: {height:.4f}m, 综合评分: {score:.4f}, 得分: {grasp.score:.4f}")

    def _print_best_grasp_info(self, best_grasp, best_angle, best_distance, best_height, best_score, object_min_z):
        """打印最优抓取信息。"""
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

    def _print_result_summary(self, best_trans_cam, best_rot_mat_cam, best_width, best_pose_base,
                              best_angle, best_distance, object_min_z):
        """打印相机与基坐标系下的最终结果。"""
        formatted_values = []
        for value in best_pose_base:
            if value >= 0:
                formatted_values.append(f"   {value:9.6f}")
            else:
                formatted_values.append(f"  {value:9.6f}")
        formatted_pose = "[" + ",".join(formatted_values) + "]"

        distance_str = f"{best_distance:.2f}" if best_distance != float('inf') else "无效"
        print("\n=== 抓取位姿结果汇总 ===")
        print("1. 相机坐标系下最优抓取位姿：")
        print(f"   - 平移 (x,y,z): {best_trans_cam.round(6)} (m)")
        print(f"   - 旋转矩阵:\n{best_rot_mat_cam.round(6)}")
        print(f"   - 抓取宽度: {best_width:.6f} (m)")
        print(f"   - 与垂直方向夹角: {best_angle:.2f}°")
        print(f"   - 距离目标中心: {distance_str}px")
        print(f"   - 抓取高度: {best_trans_cam[2]:.4f}m")
        if object_min_z is not None:
            print(f"   - 物体最低点高度: {object_min_z:.4f}m")
            print(f"   - 高度差: {(best_trans_cam[2] - object_min_z):.4f}m")
        print("\n2. 机械臂基座坐标系下目标位姿：")
        print(f"   - 位姿 [x,y,z,rx,ry,rz]: {formatted_pose}")


# 保持与原函数签名兼容的模块级函数
def run_grasp_prediction(grasp_net, color_path, depth_path, mask_path, config: Config | None = None):
    if config is None:
        config = Config.load()
    return GraspPredictor(grasp_net, config=config).predict(color_path, depth_path, mask_path, visualize=True)


def run_grasp_prediction_auto(grasp_net, color_path, depth_path, mask_path, config: Config | None = None):
    if config is None:
        config = Config.load()
    return GraspPredictor(grasp_net, config=config).predict_auto(color_path, depth_path, mask_path)
