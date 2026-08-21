# coding=utf-8
"""
眼在手上 
用采集到的图片信息和机械臂位姿信息计算相机坐标系相对于机械臂末端坐标系的旋转矩阵和平移向量

旋转矩阵的方向是  以机械臂末端为坐标系   相机  *  T  --->   end

"""

import os.path
import cv2
import numpy as np
import time
import csv
np.set_printoptions(precision=8,suppress=True)


#T中的旋转部分  包含  x  y  z三个部分

def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz@Ry@Rx   #xyz
    # R = Rx@Ry@Rz      #zyx

    return R

#完整的一个矩阵
def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]
    return H

def save_matrices_to_csv(matrices, file_name):
    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))

    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in combined_matrix:
            csv_writer.writerow(row)

def poses_save_csv(filepath):
    # 打开文本文件
    with open(filepath, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        lines = f.readlines()
    # 定义一个空列表，用于存储结果

    # 遍历每一行数据
    lines = [float(i) for line in lines for i in line.split(',')]

    matrices = []
    for i in range(0,len(lines),6):
        matrices.append(pose_to_homogeneous_matrix(lines[i:i+6]))
    # 将齐次变换矩阵列表存储到 CSV 文件中
    save_matrices_to_csv(matrices, f'./robotToolPose.csv')


def compute_T(images_path,corner_point_long,corner_point_short,corner_point_size):
    print("标定板的中长度对应的角点的个数", corner_point_long)
    print("标定板的中宽度对应的角点的个数", corner_point_short)
    print("标定板一格的长度", corner_point_size)

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # 获取标定板角点的位置
    objp = np.zeros((corner_point_long * corner_point_short, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_point_long, 0:corner_point_short].T.reshape(-1, 2)     
    # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = corner_point_size*objp

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点


    for i in range(0, 30):   #标定好的图片在images_path路径下，从0.jpg到x.jpg   一次采集的图片最多不超过30张，遍历从0.jpg到30.jpg ，选择能够读取的到的图片
        # image = f"{images_path}\\{i}.jpg"     #windows取消注释
        image = f"{images_path}/{i}.jpg"        #ubuntu下
        if os.path.exists(image):
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (corner_point_long, corner_point_short), None)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                
                # 绘制检测到的角点
                cv2.drawChessboardCorners(img, (corner_point_long,corner_point_short), corners, ret)
                # 显示图片和提示信息
                cv2.imshow(f'Image {i}', img)
                cv2.putText(img, "Press ESC for next image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.imshow(f'Image {i}', img)
                
                # 等待按键事件
                key = cv2.waitKey(0)
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
        cv2.destroyAllWindows()
    N = len(img_points)
    # 标定,得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    # print("ret:", ret)
    print("内参矩阵:\n", mtx) # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("-----------------------------------------------------")


    # 机器人末端在基坐标系下的位姿
    tool_pose = np.loadtxt("./robotToolPose.csv", delimiter=',')  #与poses_save_csv保存的名字对应上
    R_tool = []
    t_tool = []
    for i in range(int(N)):
        R_tool.append(tool_pose[0:3,4*i:4*i+3])
        t_tool.append(tool_pose[0:3,4*i+3])

    # 调用 cv2.calibrateHandEye 进行手眼标定 (方法: TSAI)
    method_tsai = cv2.CALIB_HAND_EYE_TSAI
    R_cam2gripper_tsai, t_cam2gripper_tsai = cv2.calibrateHandEye(
        R_tool, t_tool, 
        rvecs, tvecs, 
        method=method_tsai
    )
    # 输出 TSAI 方法的结果
    print("TSAI 方法计算的旋转矩阵:")
    print(R_cam2gripper_tsai)
    print("TSAI 方法计算的平移向量:")
    print(t_cam2gripper_tsai)

    # 调用 cv2.calibrateHandEye 进行手眼标定 (方法: PARK)
    method_park = cv2.CALIB_HAND_EYE_PARK
    R_cam2gripper_park, t_cam2gripper_park = cv2.calibrateHandEye(
        R_tool, t_tool, 
        rvecs, tvecs,  
        method=method_park
    )
    # 输出 PARK 方法的结果
    print("PARK 方法计算的旋转矩阵:")
    print(R_cam2gripper_park)
    print("PARK 方法计算的平移向量:")
    print(t_cam2gripper_park)

    # 调用 cv2.calibrateHandEye 进行手眼标定 (方法: HORAUD)
    method_HORAUD = cv2.CALIB_HAND_EYE_HORAUD
    R_cam2gripper_HORAUD, t_cam2gripper_HORAUD = cv2.calibrateHandEye(
        R_tool, t_tool, 
        rvecs, tvecs,  
        method=method_HORAUD
    )
    # 输出 HORAUD 方法的结果
    print("HORAUD 方法计算的旋转矩阵:")
    print(R_cam2gripper_HORAUD)
    print("HORAUD 方法计算的平移向量:")
    print(t_cam2gripper_HORAUD)    


    # 选择最优结果（这里简单地选择第一个结果作为最优）可自行选择
    return R_cam2gripper_tsai, t_cam2gripper_tsai



if __name__ == '__main__':

    images_path = "/home/zyh/ZYH_WS/eyeInHand/images" #手眼标定采集的标定版图片所在路径
    file_path = "/home/zyh/ZYH_WS/eyeInHand/images/posescopy.txt" #采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应
    corner_point_long=11     #标定板角点数量  长边
    corner_point_short=8
    corner_point_size=3        #标定板方格真实尺寸  m

    print("手眼标定采集的标定版图片所在路径", images_path)
    print("采集标定板图片时对应的机械臂末端的位姿", file_path)
    poses_save_csv(file_path)
    rotation_matrix ,translation_vector = compute_T(images_path,corner_point_long,corner_point_short,corner_point_size)
    print('默认返回tsai方法计算结果,可根据设计情况自行选择合适的矩阵和平移向量 ')
    print('rotation_matrix:')
    print(rotation_matrix)
    print('translation_vector:')
    print(translation_vector)


    """
  
    """
    
