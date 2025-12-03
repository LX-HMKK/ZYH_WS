import os
import sys
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from graspnetAPI import GraspGroup
import cv2

# 获取当前脚本的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录（父目录）
PARENT_DIR = os.path.dirname(CURRENT_DIR)
# 将上一级目录下的 `models，dataset，utils` 添加到 Python 路径
sys.path.append(os.path.join(PARENT_DIR, 'models'))
sys.path.append(os.path.join(PARENT_DIR, 'dataset'))
sys.path.append(os.path.join(PARENT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# ==================== grasp参数设置 ====================
CHECKPOINT_PATH = '/home/zyh/ZYH_WS/graspnet-baseline-main/logs/log_kn/checkpoint.tar'  # 修改为你的权重路径
NUM_POINT = 10000   #随机点云个数
NUM_VIEW = 300      # 候选抓取视角数
COLLISION_THRESH = 0.001  #碰撞检测    (m)
VOXEL_SIZE = 0.01    # 点云体素化分辨率（m）

# # 英特尔的深度相机D435i
# DEPTH_INTR = {
#     "ppx": 645.517,  # cx
#     "ppy": 377.496,  # cy
#     "fx": 908.23,   # fx
#     "fy": 908.289    # fy
# }
# 英特尔的深度相机D435
DEPTH_INTR = {
    "ppx": 644.136,  # cx
    "ppy": 354.556,  # cy
    "fx": 902.806,   # fx
    "fy": 900.776    # fy
}
DEPTH_FACTOR = 1000.0  # 深度因子，根据实际数据调整

# ==================== 网络加载 ====================
def get_net():
    """
    加载训练好的 GraspNet 模型
    """
    net = GraspNet(
        input_feature_dim=0,
        num_view=NUM_VIEW,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()  #切换网络到评估模式
    return net

#==============数据处理===============
#rgb图像的   深度图像  工作空间
def get_and_process_data(color_path,depth_path,mask_path):
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32)
        color /= 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")
    print("\n=== 尺寸验证 ===")
    print("颜色图尺寸:", color.shape[:2][::-1])
    print("深度图尺寸:", depth.shape[::-1])
    print("工作空间尺寸:", workspace_mask.shape[::-1])
    print("相机参数预设尺寸:", (1280, 720))

    #深度相机的内参
    camera = CameraInfo(
        width=1280,
        height=720,
        fx=DEPTH_INTR['fx'],
        fy=DEPTH_INTR['fy'],
        cx=DEPTH_INTR['ppx'],
        cy=DEPTH_INTR['ppy'],
        scale=DEPTH_FACTOR  #1000
    )
    #print(f"workspace_mask shape: {workspace_mask.shape}, dtype: {workspace_mask.dtype}, unique: {np.unique(workspace_mask)}")
    #print(f"depth shape: {depth.shape}, dtype: {depth.dtype}, min: {depth.min()}, max: {depth.max()}")
    #生成点云信息
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    #应用掩玛
    workspace_mask_uint8 = workspace_mask.astype(np.uint8)  # bool -> uint8
    workspace_mask_resized = cv2.resize(workspace_mask_uint8, (depth.shape[1], depth.shape[0]))
    mask = (workspace_mask_resized > 0)  # 如果需要，可以再转换回 bool
    cloud_masked =cloud[mask]
    color_masked=color[mask]

    #点云采样
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]

    #转化为open3d点云 （用于可视化）
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': cloud_sampled}

    return end_points, cloud_o3d

# ==================== 碰撞检测 ====================
def collision_detection(gg, cloud_points):
    mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=VOXEL_SIZE)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
    return gg[~collision_mask]

# ==================== 抓取位姿打印（可选） ====================
def print_grasp_poses(gg):
    print(f"\nTotal grasps after collision detection: {len(gg)}")
    for i, grasp in enumerate(gg):
        print(f"\nGrasp {i + 1}:")
        print(f"Position (x,y,z): {grasp.translation}")
        print(f"Rotation Matrix:\n{grasp.rotation_matrix}")
        print(f"Score: {grasp.score:.4f}")
        print(f"Width: {grasp.width:.4f}")

# ==================== 主函数：获取抓取预测 ====================
def run_grasp_inference(color_path, depth_path, mask_path):
    """
    主推理流程：
    1. 加载网络
    2. 处理数据并生成输入
    3. 进行抓取预测解码
    4. 碰撞检测
    5. 对抓取预测进行垂直角度筛选(仅保留接近垂直±30°的抓取)
    6. 打印/可视化抓取
    7. 返回前 50 个抓取中得分最高的抓取
    """
    #1.加载网络：
    net=get_net()

    #2.处理数据
    end_points,cloud_o3d=get_and_process_data(color_path, depth_path, mask_path)

    #3.前向推理
    with torch.no_grad():
        end_points=net(end_points)
        grasp_preds=pred_decode(end_points)

    # 4. 构造 GraspGroup 对象（这里 gg 是列表或类似列表的对象）
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())


    # 5. 碰撞检测
    if COLLISION_THRESH > 0:
        gg = collision_detection(gg, np.asarray(cloud_o3d.points))

    # 6. NMS 去重 + 按照得分排序（降序）
    gg.nms().sort_by_score()


    # ===== 新增筛选部分：对抓取预测的接近方向进行垂直角度限制 =====
    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    vertical = np.array([0, 0, -1])  # 期望抓取接近方向（垂直桌面）
    angle_threshold = np.deg2rad(30)  # 30度的弧度值
    filtered = []
    for grasp in all_grasps:
        # 抓取的接近方向取 grasp.rotation_matrix 的第三列
        approach_dir = grasp.rotation_matrix[:, 0]
        # 计算夹角：cos(angle)=dot(approach_dir, vertical)
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    if len(filtered) == 0:
        print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        filtered = all_grasps
    else:
        print(
            f"\n[DEBUG] Filtered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")

    # 对过滤后的抓取根据 score 排序（降序）
    filtered.sort(key=lambda g: g.score, reverse=True)

    # 取前50个抓取（如果少于50个，则全部使用）
    # top_grasps = filtered[:2]可以选定特定的夹抓
    top_grasps = filtered[0:1]


    # 可视化过滤后的抓取，手动转换为 Open3D 物体
    grippers = [g.to_open3d_geometry() for g in top_grasps]
    print(f"\nVisualizing top {len(top_grasps)} grasps after vertical filtering...")
    o3d.visualization.draw_geometries([cloud_o3d, *grippers])

    # 选择得分最高的抓取（filtered 列表已按得分降序排序）
    best_grasp = top_grasps[0]
    best_translation = best_grasp.translation
    best_rotation = best_grasp.rotation_matrix
    best_width = best_grasp.width

    # 新增：返回前10个候选抓取列表
    return (best_translation, best_rotation, best_width, top_grasps)


# ==================== 如果希望直接在此脚本中测试，可保留 main ====================
if __name__ == '__main__':
    # 示例：使用文件路径
    color_img_path = '/home/zyh/ZYH_WS/graspnet-baseline-main/zyh_code/test2/aligned_color_0000.png'
    depth_img_path = '/home/zyh/ZYH_WS/graspnet-baseline-main/zyh_code/test2/aligned_depth_0000.png'
    mask_img_path = '/home/zyh/ZYH_WS/graspnet-baseline-main/zyh_code/test2/sam_mask.png'

     # 现在会返回四个值：最优参数 + 前10抓取列表
    t, R_mat, w, top_grasps = run_grasp_inference(color_img_path, depth_img_path, mask_img_path)
    
    # 打印最优抓取
    print("\n=== 最优抓取信息 ===")
    print(f"Translation= {t}")
    print(f"Rotation=\n{R_mat}")
    print(f"宽度: {w:.4f}")
    print(f"得分: {top_grasps[0].score:.4f}")  # 添加最优得分
 
    # 新增：打印其他候选抓取
    print("\n=== 其他候选抓取信息 (前10排序) ===")
    for i, grasp in enumerate(top_grasps[1:], 1):  # 从第二个开始遍历
        print(f"\n候选抓取 #{i+1}:")
        print(f"位置 (x,y,z): {grasp.translation}")
        print(f"旋转矩阵:\n{grasp.rotation_matrix}")
        print(f"宽度: {grasp.width:.4f}")
        print(f"得分: {grasp.score:.4f}")
        print(f"与最优抓取的分数差: {top_grasps[0].score - grasp.score:.4f}")