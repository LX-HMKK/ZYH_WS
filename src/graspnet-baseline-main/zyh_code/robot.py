import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_new(
        grasp_translation,  # GraspNet 输出的平移 (相机坐标系下)
        grasp_rotation_mat,  # GraspNet 输出的旋转矩阵 (相机坐标系下, 3x3)
        current_ee_pose,  # 机械臂当前末端在基座坐标系下的位姿 [x, y, z, rx, ry, rz]
        handeye_rot,  # 手眼标定旋转矩阵 (末端→相机)：相机相对于末端的旋转
        handeye_trans,  # 手眼标定平移向量 (末端→相机)：相机原点相对于末端的平移
        tool_rot,  # 工具坐标系旋转矩阵 (末端→工具)：工具相对于末端法兰的旋转
        tool_trans,  # 工具坐标系平移向量 (末端→工具)：工具作用点相对于末端法兰的平移
        gripper_length=0.20  # 夹爪长度（若工具坐标系已包含，可省略）
):
    # 1. 坐标系对齐矩阵（GraspNet输出与机械臂坐标系对齐）
    R_adjust = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ], dtype=np.float32)
    T_align = np.eye(4, dtype=float)
    T_align[:3, :3] = R_adjust
    
    # 2. 抓取位姿→相机坐标系（应用对齐）
    T_grasp2cam = np.eye(4)
    T_grasp2cam[:3, :3] = grasp_rotation_mat
    T_grasp2cam[:3, 3] = grasp_translation
    T_grasp2cam = T_align @ T_grasp2cam  # 先对齐抓取坐标系，再转换到相机（修正顺序）
    
    # 3. 手眼标定：相机→末端坐标系（修正手眼变换的逆）
    # 手眼参数原定义为“末端→相机”，需转换为“相机→末端”
    R_cam2end = handeye_rot.T  # 旋转矩阵逆=转置（正交矩阵性质）
    t_cam2end = -R_cam2end @ handeye_trans  # 平移向量修正
    T_cam2end = np.eye(4)
    T_cam2end[:3, :3] = R_cam2end
    T_cam2end[:3, 3] = t_cam2end
    
    # 4. 末端→基座坐标系（基于当前末端位姿）
    x, y, z, rx, ry, rz = current_ee_pose
    R_end2base = R.from_euler('XYZ', [rx, ry, rz]).as_matrix()
    T_end2base = np.eye(4)
    T_end2base[:3, :3] = R_end2base
    T_end2base[:3, 3] = [x, y, z]
    
    # 5. 计算抓取位姿在基座坐标系下的变换（工具作用点的目标位姿）
    T_base2grasp = T_end2base @ T_cam2end @ T_grasp2cam  # 基座→工具作用点（抓取点）
    
    # 6. 工具坐标系：构造“末端→工具”的变换矩阵 T_tool
    T_tool = np.eye(4)
    T_tool[:3, :3] = tool_rot  # 工具相对于末端的旋转
    T_tool[:3, 3] = tool_trans  # 工具作用点相对于末端的平移
    
    # 7. 计算 T_tool 的逆矩阵（工具→末端），用于从工具位姿反推末端法兰位姿
    T_tool_inv = np.eye(4)
    T_tool_inv[:3, :3] = tool_rot.T  # 旋转逆=转置
    T_tool_inv[:3, 3] = -tool_rot.T @ tool_trans  # 平移逆修正
    
    # 8. 核心：计算末端法兰需要到达的位姿（基座坐标系下）
    # 逻辑：末端法兰位姿 = 工具作用点目标位姿 × 工具坐标系逆变换
    T_base2end_final = T_base2grasp @ T_tool_inv
    
    # 9. 提取最终位姿（平移+旋转）
    final_trans = T_base2end_final[:3, 3]
    final_rot = R.from_matrix(T_base2end_final[:3, :3])
    base_rx, base_ry, base_rz = final_rot.as_euler('XYZ')
    
    return np.concatenate([final_trans, [base_rx, base_ry, base_rz]])


# 测试用例（加入工具坐标系参数）
if __name__ == '__main__':
    # 1. GraspNet输出（相机坐标系下的抓取位姿）
    grasp_translation = [0.03889914, -0.0461118, 0.409]
    grasp_rotation_mat = np.array([
        [-0.15578863, 0.90573424, 0.39417678],
        [-0.33291125, -0.423846, 0.84233284],
        [0.93, 0.0, 0.3675595]
    ])
    
    # 2. 机械臂当前末端位姿（基座坐标系下，[x,y,z,rx,ry,rz]）
    current_ee_pose = [
        0.1080898989423529, 0.3258393976566775, -0.28685235674255904,
        7.705391880108516e-12, -6.429548966107223e-07, 3.141556700929793
    ]
    
    # 3. 手眼标定参数（末端→相机）
    handeye_rot = np.array([
        [0.99975725, 0.00723614, -0.02081039],
        [-0.00707294, 0.99994374, 0.00790557],
        [0.02086643, -0.00775646, 0.99975218]
    ])
    handeye_trans = np.array([-0.03426620597, -0.08995216606, -0.0148644684])
    
    # 4. 工具坐标系参数（末端→工具：夹爪相对于末端法兰的偏移）
    # 示例：假设工具（夹爪）相对于末端法兰的旋转为0（与末端同向），
    # 平移为[0,0,-0.15]（沿末端Z轴负方向15cm，即夹爪长度补偿）
    tool_rot = np.eye(3)  # 工具与末端法兰旋转一致
    tool_trans = np.array([0.0, 0.0, -0.15])  # 工具作用点（指尖）在末端坐标系下的位置
    
    # 转换计算
    result = convert_new(
        grasp_translation,
        grasp_rotation_mat,
        current_ee_pose,
        handeye_rot,
        handeye_trans,
        tool_rot=tool_rot,
        tool_trans=tool_trans,
        gripper_length=0.195  # 若工具坐标系已包含长度，可设为0
    )
    
    print("考虑工具坐标系后，末端法兰的目标位姿:", result)
