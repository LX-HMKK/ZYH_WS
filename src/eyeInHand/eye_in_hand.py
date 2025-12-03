import cv2
import numpy as np
import transforms3d
import glob
import re
import os
def euler_to_rotation_matrix(rx, ry, rz, unit='deg'):  # rx, ry, rz是欧拉角，单位是度
    '''
    将欧拉角转换为旋转矩阵：R = Rz * Ry * Rx
    :param rx: x轴旋转角度
    :param ry: y轴旋转角度
    :param rz: z轴旋转角度
    :param unit: 角度单位，'deg'表示角度，'rad'表示弧度
    :return: 旋转矩阵
    '''
    if unit == 'deg':
        # 把角度转换为弧度
        rx = np.radians(rx)
        ry = np.radians(ry)
        rz = np.radians(rz)

    # 计算旋转矩阵Rz 、 Ry 、 Rx
    Rx = transforms3d.axangles.axangle2mat([1, 0, 0], rx)
    Ry = transforms3d.axangles.axangle2mat([0, 1, 0], ry)
    Rz = transforms3d.axangles.axangle2mat([0, 0, 1], rz)

    # 计算旋转矩阵R = Rz * Ry * Rx
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    return rotation_matrix


# 从末端位姿中提取变换矩阵
def pose_vector_to_end2base_transforms(pose_vector):
    R_end2bases = []
    t_end2bases = []

    # 迭代遍历每个位姿态的旋转矩阵和平移向量

    for pose_vector in pose_vector:
        R_end2base = euler_to_rotation_matrix(pose_vector[3], pose_vector[4], pose_vector[5])
        t_end2base = pose_vector[:3]

        # 提取旋转矩阵和平移向量
        R_end2bases.append(R_end2base)
        t_end2bases.append(t_end2base)
    return  R_end2bases, t_end2bases


# 自定义排序函数：提取文件名中的数字部分

def natural_sort_key(path):

    filename = os.path.basename(path)

    num = re.search(r"(\d+)", filename).group(1)  # 提取数字部分

    return int(num)  # 转换为整数排序
# 按数字顺序排序

    # 所有图像的路径
images = glob.glob("/home/zyh/ZYH_WS/eyeInHand/images/*.jpg")
    #注意排序是按照字符串的顺序  
sorted_paths = sorted(images, key=natural_sort_key)
# 输入位姿数据，注意欧拉角是角度还是弧度  主要不加入科学计数法
pose_vectors = np.array([
[108.182,296.141,-117.455,3.109,-9.756,-107.869],
[134.404,210.005,-165.789,-0.325,-14.18,-108.04],
[137.627,204.624,-187.535,-0.946,-15.106,-107.607],
[84.23,218.64,-165.119,8.465,-13.589,-107.53],
[83.839,261.931,-305.13,8.16,-12.672,-107.454],
[89.868,338.597,-181.648,3.23,1.196,-107.697],
[148.787,249.195,-277.689,-2.305,-15.808,-108.32],
[31.699,295.752,-227.027,1.473,-9.892,-90.714],
[51.111,301.622,-166.614,-2.586,-9.06,-90.597],
[133.794,201.533,-258.156,8.449,-12.923,-120.958],
[155.118,292.95,-150.343,5.523,-3.907,-120.541],
[144.82,270.204,-143.667,8.492,-10.37,-120.771],
[119.961,217.325,-234.341,11.356,-12.169,-120.814],
[137.472,237.709,-193.179,0.285,-17.123,-87.346],
[74.476,238.16,-232.636,0.025,-6.532,-69.157],
[74.235,238.343,-255.525,17.485,-6.748,-148.174],
[90.333,244.59,-164.703,12.42,-12.135,-134.537]
                            ])
square_size = 30.0   #标定板格子的长度
pattern_size = (11, 8)   # 


# 导入相机内参和畸变参数
# 焦距 fx, fy, 光心 cx, cy
# 畸变系数 k1, k2

fx, fy, cx, cy = 894.0968168, 893.9089673, 644.51383224, 357.72195187
k1, k2 =  0.13654039, -0.30612869
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float64)   # K为相机内参矩阵
dist_coeffs = np.array([k1, k2, 0, 0], dtype=np.float64)   # 畸变系数

if __name__ == '__main__':

    # 求解手眼标定
    # R_end2bases：机械臂末端相对于机械臂基座的旋转矩阵
    # t_end2bases：机械臂末端相对于机械臂基座的平移向量
    R_end2bases, t_end2bases = pose_vector_to_end2base_transforms(pose_vectors)
    # print("R_end2bases:")
    # print(R_end2bases)
    # print("t_end2bases:")
    # print(t_end2bases)


    # 准备位姿数据
    obj_points = []  # 用于保存世界坐标系中的三维点
    img_points = []  # 用于保存图像平面上的二维点

    # 创建棋盘格3D坐标
    objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
# 迭代处理图像
det_success_num = 0  # 用于保存检测成功的图像数量
for image in sorted_paths:
    img = cv2.imread(image)  # 读取图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RGB图像转换为灰度图像

    # 棋盘格检测
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    #棋盘  角点坐标
    if ret:
        det_success_num += 1
        # 如果成功检测到棋盘格，添加图像平面上的二维点和世界坐标系中的三维点到列表
        obj_points.append(objp)
        img_points.append(corners)

        # 绘制并显示角点
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.namedWindow('img')
        cv2.imshow('img', img)
        cv2.waitKey(50)

print("检测到正确的棋盘个数" , det_success_num)
cv2.destroyAllWindows()
# 求解标定板位姿
R_board2cameras = []  # 用于保存旋转矩阵
t_board2cameras = []  # 用于保存平移向量

for i in range(det_success_num):
    # rvec：标定板相对于相机坐标系的旋转向量
    # t_board2camera：标定板相对于相机坐标系的平移向量
    ret, rvec, t_board2camera = cv2.solvePnP(obj_points[i], img_points[i], K, dist_coeffs)

    # 将旋转向量(rvec)转换为旋转矩阵
    # R：标定板相对于相机坐标系的旋转矩阵
    R_board2camera, _ = cv2.Rodrigues(rvec)  # 输出：R为旋转矩阵和旋转向量的关系  输入：rvec为旋转向量

    # 将标定板相对于相机坐标系的旋转矩阵和平移向量保存到列表
    R_board2cameras.append(R_board2camera)
    t_board2cameras.append(t_board2camera)

print("R_board2cameras")
print(R_board2cameras)
print("t_board2cameras")
print(t_board2cameras)

# 求解手眼标定
# R_end2bases：机械臂末端相对于机械臂基座的旋转矩阵
# t_end2bases：机械臂末端相对于机械臂基座的平移向量
R_end2bases, t_end2bases = pose_vector_to_end2base_transforms(pose_vectors)

# R_camera2end：相机相对于机械臂末端的旋转矩阵
# t_camera2end：相机相对于机械臂末端的平移向量
R_camera2end, t_camera2end = cv2.calibrateHandEye(R_end2bases, t_end2bases,
                                                  R_board2cameras, t_board2cameras,
                                                  method=cv2.CALIB_HAND_EYE_TSAI)

# 将旋转矩阵和平移向量组合成齐次位姿矩阵
T_camera2end = np.eye(4)
T_camera2end[:3, :3] = R_camera2end
T_camera2end[:3, 3] = t_camera2end.reshape(3)
# 输出相机相对于机械臂末端的旋转矩阵和平移向量
print("Camera to end rotation matrix:")
print(R_camera2end)
print("Camera to end translation vector:")
print(t_camera2end)

# 输出相机相对于机械臂末端的位姿矩阵
print("Camera to end pose matrix:")
np.set_printoptions(suppress=True)  # suppress参数用于禁用科学计数法
print(T_camera2end)








