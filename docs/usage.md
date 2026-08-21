# 机械臂上位机使用说明

本文档介绍 RoboArm-Vision 的安装、配置、启动与常见问题排查。

## 1. 环境要求

- **操作系统**：Ubuntu 22.04（推荐，用于 RealSense、ROS 2、录音等功能）
- **ROS 2**：Humble Hawksbill
- **Python**：3.10+
- **CUDA**：11.8+（推荐，用于深度学习推理）
- **硬件**：
  - 六自由度机械臂 + 下位机控制器
  - Intel RealSense D435/D435i
  - 达妙电机夹爪
  - 麦克风（可选，用于语音交互）

## 2. 安装步骤

### 2.1 安装 ROS 2 依赖

```bash
sudo apt update
sudo apt install -y ros-humble-realsense2-camera ros-humble-cv-bridge
```

### 2.2 克隆仓库并设置环境变量

```bash
git clone <repo-url> /home/zyh/ZYH_WS
cd /home/zyh/ZYH_WS

# 将以下行添加到 ~/.bashrc
export ROBOARM_WS=/home/zyh/ZYH_WS
source /opt/ros/humble/setup.bash
```

### 2.3 安装 Python 依赖

建议使用 Conda 环境：

```bash
conda create -n grasp python=3.10
conda activate grasp

# 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics segment-anything open3d pyrealsense2 opencv-python scipy numpy pyyaml
```

### 2.4 安装 GraspNet 原生算子

`grasp_pipeline` 依赖 `tools/graspnet_baseline` 中的 GraspNet 基线网络、原生 CUDA 算子与评估库：

```text
tools/graspnet_baseline/
├── pointnet2/          # PointNet++ CUDA 算子
├── knn/                # KNN CUDA 算子
├── graspnetAPI-master/ # GraspNet API（可 pip install .）
├── models/             # GraspNet 网络定义
├── utils/              # GraspNet 工具函数
└── requirements.txt
```

首次运行前必须安装原生算子：

```bash
cd $ROBOARM_WS/tools/graspnet_baseline
pip install -r requirements.txt
cd pointnet2 && python setup.py install
cd ../knn && python setup.py install
cd ../graspnetAPI-master && pip install .
```

### 2.5 安装 ROS 2 Python 依赖

```bash
pip install rosdep2 setuptools
rosdep install --from-paths src --ignore-src -y
```

## 3. 准备模型权重

将模型文件放入 `assets/` 目录（已加入 `.gitignore`，不提交到 Git）：

```text
assets/
├── all.pt                 # YOLOv8 检测模型
├── sam_vit_b_01ec64.pth   # SAM 分割模型
└── checkpoint.tar         # GraspNet 抓取模型
```

> 注意：仓库不提供模型权重，请自行下载或从比赛主办方获取。

## 4. 配置 API Key

```bash
cd $ROBOARM_WS
cp config/api_keys.yaml.example config/api_keys.yaml
# 编辑 config/api_keys.yaml，填入真实 key
```

示例：

```yaml
zhipu_api_key: "your-real-key-here"
```

## 5. 手眼标定

### 5.1 采集棋盘格图像

```bash
cd $ROBOARM_WS/tools/eyeInHand
python3 保存RGB图像.py
```

按提示采集约 30 张不同位姿的棋盘格图像，保存到 `eyeInHand/images/`。

### 5.2 计算手眼矩阵

```bash
python3 eye_in_hand.py
```

输出的 `handeye_rot` 和 `handeye_trans` 需要填入 `src/vision_grasp/config/grasp_config.yaml`。

### 5.3 更新配置

编辑 `src/vision_grasp/config/grasp_config.yaml`：

```yaml
current_ee_pose: [x, y, z, rx, ry, rz]  # 当前末端位姿
handeye_rot:
  - [r11, r12, r13]
  - [r21, r22, r23]
  - [r31, r32, r33]
handeye_trans: [tx, ty, tz]
```

所有单位均为米 / 弧度。

## 6. 构建工作空间

```bash
cd $ROBOARM_WS
conda activate grasp
python -m colcon build --symlink-install
source install/setup.bash
```

首次构建或修改接口包后，务必重新 source：

```bash
source install/setup.bash
```

## 7. 启动系统

### 7.1 一键启动

```bash
ros2 launch arm_bringup robot_arm.launch.py
```

### 7.2 分别启动节点

每个节点一个终端，均先执行：

```bash
cd $ROBOARM_WS
source install/setup.bash
```

终端 1：视觉 / 抓取检测

```bash
ros2 run vision_grasp grasp_node
```

终端 2：上位机通信 + 夹爪控制

```bash
ros2 run arm_control io_node
```

终端 3：运动状态机

```bash
ros2 run arm_control motion_node
```

终端 4（可选）：大语言模型交互

```bash
ros2 run llm_voice llm_node
```

## 8. 无硬件测试

### 8.1 测试运动状态机

发布模拟的 `GraspResult`：

```bash
ros2 topic pub /grasp_result arm_interfaces/msg/GraspResult "{
  trans_cam: [0.0,0.0,0.0],
  rot_cam_flat: [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
  width: 0.05,
  score: 0.95,
  pos_base: [-25.928, -427.63, -27.168],
  euler_base: [21.394, 40.531, 10.292],
  cls_name: 'banana',
  stamp: {sec: 0, nanonsec: 0}
}" --once
```

触发下一次检测：

```bash
ros2 topic pub -1 /robot_status std_msgs/msg/String "data: 'have backed'"
```

### 8.2 测试 LLM 服务

```bash
ros2 service call /llm/ask_text llm_voice/srv/AskText "{question: '你好'}"
```

## 9. 常用调试命令

### 9.1 查看话题

```bash
ros2 topic list
ros2 topic echo /grasp_result
ros2 topic echo /RobotInfo
ros2 topic echo /robot_status
```

### 9.2 查看节点图

```bash
ros2 run rqt_graph rqt_graph
```

### 9.3 查看服务

```bash
ros2 service list
```

## 10. 配置调参

### 10.1 抓取检测参数

编辑 `src/vision_grasp/config/grasp_config.yaml`：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `camera_res` | RealSense 分辨率 | `[1280, 720]` |
| `depth_intr` | 深度相机内参 | 根据实际标定 |
| `num_point` | 点云采样数 | 10000 |
| `collision_thresh` | 碰撞检测阈值 | 0.001 m |
| `mask_choice` | 0=SAM 掩码，1=YOLO 扩展掩码 | 0 |
| `gripper_length` | 夹爪长度补偿 | -0.14 m |

### 10.2 运动参数

编辑 `src/arm_control/config/motion_config.yaml`：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `home_xyz` / `home_euler` | home 位姿 | 根据实际机械臂 |
| `place_xyz` / `place_euler` | 放置位姿 | 根据实际场景 |
| `safe_z_height` | 提升安全高度 | 根据实际机械臂 |
| `position_tolerance` | 位置容差 | 5 mm |
| `rotation_tolerance` | 旋转容差 | 0.5 deg |
| `step_timeout` | 单步超时 | 10 s |
| `class_compensation` | 类别 Z 补偿 | 根据物体调整 |

### 10.3 TCP 与夹爪参数

通过 ROS 参数覆盖 `io_node` 默认值：

```bash
ros2 run arm_control io_node \
  --ros-args \
  -p server_host:="172.16.26.125" \
  -p server_port:=10001 \
  -p allowed_client:="172.16.26.126" \
  -p gripper_port:="/dev/ttyACM0"
```

## 11. 常见问题

### 11.1 RealSense 初始化失败

- 检查相机是否被其他进程占用
- 检查 USB 带宽，尝试更换 USB 3.0 端口
- `grasp_node` 通过 `grasp_pipeline.core.camera_driver.RealSenseCamera` 自动重试启动，运行中帧超时时也会自动重启（最多 3 次）

### 11.2 GraspNet 未预测到抓取

- 检查 `assets/checkpoint.tar` 是否存在
- 检查点云是否为空（深度图、掩码是否正确）
- 降低 `collision_thresh` 或 `voxel_size`

### 11.3 坐标转换结果明显偏差

- 重新执行手眼标定
- 检查 `current_ee_pose` 是否与实际一致
- 检查 `gripper_length` 符号和数值

### 11.4 运动状态机一直不推进

- 确认 `/RobotInfo` 有数据：`ros2 topic echo /RobotInfo`
- 检查 `position_tolerance` 和 `rotation_tolerance` 是否过严
- 检查 `step_timeout` 是否过短

### 11.5 夹爪无响应

- 检查串口权限：`sudo usermod -aG dialout $USER`
- 确认 `gripper_port` 参数正确
- 查看 `io_node` 日志中的初始化错误

### 11.6 LLM 语音交互在 Windows 下不可用

录音功能依赖 Linux 的 `arecord`，Windows 下调试时仅文本服务可用。

## 12. 硬编码配置清单

以下默认值已通过 ROS 参数或 YAML 配置文件暴露，优先修改配置而非源码：

| 位置 | 说明 |
|------|------|
| `src/vision_grasp/vision_grasp/grasp_node.py` | 通过 `ROBOARM_WS` 解析 `src/` 根目录 |
| `src/arm_control/arm_control/io_node.py` | 默认服务器 IP、允许客户端 IP、夹爪串口可通过 ROS 参数覆盖 |
| `src/arm_control/arm_control/motion_node.py` | 默认读取 `src/arm_control/config/motion_config.yaml`，可通过 `motion_config_path` 参数修改 |
| `src/llm_voice/llm_voice/llm_node.py` | 默认读取 `config/api_keys.yaml`，可通过 `api_key_path` 参数修改 |
| `src/grasp_pipeline/config.py` | 所有模型路径、相机内参、手眼参数、当前末端位姿从 `src/vision_grasp/config/grasp_config.yaml` 加载 |
| `tools/eyeInHand/eye_in_hand.py` | 图像目录、棋盘格参数、相机内参硬编码在文件内 |

## 13. 夹爪驱动细节

夹爪控制代码已内联到 `arm_control` 包中：

- `arm_control/arm_control/gripper_can.py` —— 底层达妙电机 CAN/串口协议（原 `tools/Gloria-M-SDK-1.0.0/motor/DM_CAN.py`）。
- `arm_control/arm_control/gripper_driver.py` —— 夹爪初始化、打开、闭合的高层接口，暴露：
  - `init_gripper(port)`
  - `open_gripper()`
  - `close_gripper()`

`io_node` 直接通过 `from arm_control import gripper_driver` 加载，无需 `sys.path.append`。

## 14. 开发规范

- 新增 ROS 功能包放在 `src/`
- 新增非 ROS 工具/脚本放在 `tools/`
- 模型权重放在 `assets/`，不要提交到 Git
- 提交信息使用 Angular 格式，中文简述：`refactor(repo): 重命名功能包`

## 13. 参考文档

- [系统设计文档](./architecture.md)
- [项目 README](../README.md)
- [Claude Code 工作指南](../CLAUDE.md)
