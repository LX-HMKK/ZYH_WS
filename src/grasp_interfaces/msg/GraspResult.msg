# 相机坐标系下的抓取位姿
float64[3] trans_cam
float64[9] rot_cam_flat      # 按行展开 3x3
float64 width                # 单位 m
float64 score                # GraspNet 置信度

# 机械臂基座坐标系下的目标位姿（XYZ+XYZ 欧拉角）
float64[3] pos_base
float64[3] euler_base        # 单位 rad

# 物体类别名
string cls_name

# 时间戳
builtin_interfaces/Time stamp 