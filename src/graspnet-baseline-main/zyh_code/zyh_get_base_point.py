import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_new(
        grasp_translation,  # GraspNet 输出的平移 (相机坐标系下)
        grasp_rotation_mat,  # GraspNet 输出的旋转矩阵 (相机坐标系下, 3x3)
        current_ee_pose,  # 机械臂当前末端在基座坐标系下的位姿 [x, y, z, rx, ry, rz]
        handeye_rot,  # 手眼标定旋转矩阵 (末端→相机)
        handeye_trans,  # 手眼标定平移向量 (末端→相机)
):
    # 坐标系对齐矩阵（保持原逻辑）
    R_adjust = np.array([
        [0, 0, 1],  # 将X轴旋转到Z轴
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

    # 核心修改：基座到抓取的完整变换链
    # 修正变换顺序：基座→末端→相机→抓取
    T_base2grasp = T_end2base @ T_cam2end @ T_grasp2cam
    
    # 提取最终位姿（旋转+平移）
    final_trans = T_base2grasp[:3, 3]
    final_rot = R.from_matrix(T_base2grasp[:3, :3])
    base_rx, base_ry, base_rz = final_rot.as_euler('XYZ')
    

    return np.concatenate([final_trans, [base_rx, base_ry, base_rz]])

# 测试用例
if __name__ == '__main__':
    Translation= [ 0.13073827, -0.05534238 , 0.439     ]
    Rotation=[
    [-0.11492483,  0.9931735,  -0.01996705],
    [ 0.36639848 , 0.06106357,  0.92845213],
    [ 0.92333335 , 0.09938631, -0.370915  ]]

    end_positions = [0.10809402105592679, 0.3258393975737154, -0.09185500000868108, 2.5413909196778034e-26, -1.2627174896584495e-05, 3.1415686851497933]
    
    cam2robo_Rotation = np.array([
        [ 0.9999615  ,-0.00144533 ,-0.00865486],
        [ 0.00140031 , 0.99998547 ,-0.00520601],
        [ 0.00866225 , 0.00519369 ,0.99994899]
        ])
    cam2robo_Translation = np.array([ -0.03347189506, -0.08979667328,-0.21301463981])
    
    result = convert_new(
        Translation,
        Rotation,
        end_positions,
        cam2robo_Rotation,
        cam2robo_Translation,
    )
    
    print("最终机械臂到达的位姿:", result)