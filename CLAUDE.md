# CLAUDE.md

本文件为 Claude Code（claude.ai/code）提供在本仓库中工作时的指导。

## 项目概述

本项目是 2025 年机器人学课程的比赛作品——“机械臂上位机程序”。它集成了六自由度机械臂、Intel RealSense 视觉、基于深度学习的抓取检测以及大语言模型语音交互。

本仓库现在按标准 ROS 2 workspace 组织：

- `src/` —— 仅包含 ROS 2 功能包。
- `assets/` —— 模型权重等大文件（已加入 `.gitignore`，不提交到 Git）。
- `tools/` —— 非 ROS 的工具/库/脚本。
- `config/` —— 共享运行时配置（API key 等）。
- `docs/` —— 详细设计文档与使用说明。

主要子系统：

- `src/robot_arm_bringup/` —— 启动文件（launch）。
- `src/robot_arm_control/` —— 下位机 TCP 通信、夹爪控制、抓取-放置状态机。
- `src/vision_grasp/` —— RealSense 视觉节点，负责调用 `grasp_pipeline` 并发布抓取结果。
- `src/grasp_pipeline/` —— YOLO + SAM + GraspNet 抓取检测流水线及工具类。
- `src/llm_voice/` —— 智谱 AI 大语言模型语音/文本交互。
- `src/robot_arm_interfaces/` —— 自定义消息/服务定义。
- `tools/graspnet_baseline/` —— GraspNet 基线网络、原生算子与评估库。
- `tools/eyeInHand/` —— 基于棋盘格的手眼标定。

## 参考文档

- `docs/architecture.md` —— 系统设计文档（架构、数据流、坐标变换、状态机）。
- `docs/usage.md` —— 使用说明（安装、配置、启动、调试、常见问题）。
- `README.md` —— 项目简介与快速开始。

所有 ROS 节点通过 `ROBOARM_WS` 环境变量定位仓库根目录，未设置时默认回退到 `/home/zyh/ZYH_WS`。

```python
def get_workspace_root() -> str:
    return os.environ.get("ROBOARM_WS", "/home/zyh/ZYH_WS")
```

典型路径：

- 模型权重：`{ROBOARM_WS}/assets/all.pt`、`{ROBOARM_WS}/assets/sam_vit_b_01ec64.pth`
- GraspNet 基线根目录：`{ROBOARM_WS}/tools/graspnet_baseline`
- 抓取检测流水线：`{ROBOARM_WS}/src/grasp_pipeline`
- 视觉/抓取节点配置：`{ROBOARM_WS}/src/vision_grasp/config/grasp_config.yaml`
- 运动配置：`{ROBOARM_WS}/src/robot_arm_control/config/motion_config.yaml`
- API key：`{ROBOARM_WS}/config/api_keys.yaml`

## 构建与运行命令

### ROS 2 工作空间

在仓库根目录执行：

```bash
export ROBOARM_WS=/home/zyh/ZYH_WS
cd $ROBOARM_WS
conda activate grasp
python -m colcon build --symlink-install
```

加载构建环境：

```bash
source install/setup.bash
```

启动整个系统：

```bash
ros2 launch robot_arm_bringup robot_arm.launch.py
```

或手动启动三个核心节点（每个节点一个终端）：

```bash
ros2 run vision_grasp grasp_node      # 视觉 / 抓取检测
ros2 run robot_arm_control io_node    # 上位机与下位机 TCP 通信 + 夹爪控制
ros2 run robot_arm_control motion_node # 运动状态机
```

`llm_voice` 节点需要单独启动。API key 从 `config/api_keys.yaml` 读取（该文件已加入 `.gitignore`，仓库中只保留模板 `config/api_keys.yaml.example`）：

```bash
# 复制模板并填入真实 key
cp config/api_keys.yaml.example config/api_keys.yaml
# 编辑 config/api_keys.yaml 后启动
ros2 run llm_voice llm_node
```

`llm_node` 也支持通过 ROS 参数指定 key 文件路径：

```bash
ros2 run llm_voice llm_node --ros-args -p api_key_path:=/path/to/api_keys.yaml
```

### 无硬件时测试抓取流程

发布一个模拟的 `GraspResult`：

```bash
ros2 topic pub /grasp_result robot_arm_interfaces/msg/GraspResult "{
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

触发运动测试状态机：

```bash
ros2 topic pub -1 /robot_status std_msgs/msg/String "data: 'have backed'"
```

### 夹爪驱动（`src/robot_arm_control/robot_arm_control/gripper_driver.py`）

夹爪控制代码已内联到 `robot_arm_control` 包中，不再依赖外部 SDK 目录：

- `gripper_can.py` —— 底层达妙电机 CAN/串口协议（原 `tools/Gloria-M-SDK-1.0.0/motor/DM_CAN.py`）。
- `gripper_driver.py` —— 夹爪初始化、打开、闭合的高层接口（原 `tools/Gloria-M-SDK-1.0.0/motor/DM_Motor_Test.py`）。

`io_node.py` 直接通过 `from robot_arm_control import gripper_driver` 加载，无需 `sys.path.append`。

### 手眼标定（`tools/eyeInHand`）

```bash
cd tools/eyeInHand
python3 保存RGB图像.py   # 从 RealSense 采集约 30 张棋盘格图像
python3 eye_in_hand.py   # 计算相机到机械臂末端的变换
```

`eye_in_hand.py` 中的硬编码路径与参数：

- 图像路径：`/home/zyh/ZYH_WS/eyeInHand/images/*.jpg`
- 棋盘格参数：`square_size = 30.0`，`pattern_size = (11, 8)`
- 相机内参与畸变系数在文件底部硬编码。

### GraspNet 基线（`tools/graspnet_baseline`）

`tools/graspnet_baseline` 保留 GraspNet 基线网络的原生算子、模型定义与评估库，是 `grasp_pipeline` 的底层依赖：

```
tools/graspnet_baseline/
├── pointnet2/          # PointNet++ CUDA 算子
├── knn/                # KNN CUDA 算子
├── graspnetAPI-master/ # GraspNet API（可 pip install .）
├── models/             # GraspNet 网络定义
├── utils/              # GraspNet 工具函数
└── requirements.txt
```

安装原生算子（首次运行前必需）：

```bash
cd tools/graspnet_baseline
pip install -r requirements.txt
cd pointnet2 && python setup.py install
cd ../knn && python setup.py install
cd ../graspnetAPI-master && pip install .
```

### 抓取检测流水线（`src/grasp_pipeline`）

`src/grasp_pipeline` 是项目定制的视觉/抓取检测流水线，按职责拆分为多个工具类模块：

```
src/grasp_pipeline/
├── __init__.py
├── package.xml
├── setup.py
├── config.py              # Config 配置类
├── core/
│   ├── model_manager.py   # ModelManager：加载 YOLO / SAM / GraspNet
│   ├── frame_processor.py # FrameProcessor：RealSense 帧对齐
│   ├── object_segmentor.py# ObjectSegmentor：YOLO + SAM 掩码生成
│   ├── grasp_predictor.py # GraspPredictor：GraspNet 推理与坐标转换
│   └── data_processor.py  # DataProcessor：点云预处理与碰撞检测
├── transforms/
│   └── coordinate_transformer.py # CoordinateTransformer：相机→基坐标系
└── utils/
    └── vision_utils.py    # IoU / NMS 等视觉辅助函数
```

`src/vision_grasp/vision_grasp/grasp_node.py` 通过 `sys.path.append("{ROBOARM_WS}/src")` 导入上述模块。

## 高层架构

### ROS 2 节点关系图

运行时由三个核心节点和一个可选节点组成：

1. **`vision_grasp/grasp_node`**（`src/vision_grasp/vision_grasp/grasp_node.py`）
   - 采集 RealSense 帧，依次运行 YOLOv8 检测、SAM 分割、GraspNet 抓取预测，将最优抓取从相机坐标系转换到机械臂基坐标系，并向 `/grasp_result` 发布 `robot_arm_interfaces/GraspResult`。
   - 订阅 `/robot_status`，收到 `"have backed"` 后触发下一次检测。
   - 帧超时时会自动重启相机（最多 3 次）。
   - 临时图像使用 `tempfile.NamedTemporaryFile`，不再固定写入 `/tmp`。
   - 通过 `sys.path.append("{ROBOARM_WS}/src")` 导入 `grasp_pipeline.config`、`grasp_pipeline.core.*`、`grasp_pipeline.transforms.*`。

2. **`robot_arm_control/io_node`**（`src/robot_arm_control/robot_arm_control/io_node.py`）
   - 由三个专注的类组成：
     - `RobotTcpServer`：非阻塞 TCP 服务器，监听下位机连接，维护连接池，接收/发送数据。
     - `GripperController`：封装 `src/robot_arm_control/robot_arm_control/gripper_driver.py` 的夹爪初始化、打开、闭合，支持失败后定时重连。
     - `CodroidIONode`：ROS 节点，聚合上述两个子系统；订阅 `RobotMove` 并转发给所有下位机；订阅 `GripperControl` 控制夹爪；发布 `RobotInfo`。
   - 服务器 IP、端口、允许客户端 IP、夹爪串口均通过 ROS 参数暴露。

3. **`robot_arm_control/motion_node`**（`src/robot_arm_control/robot_arm_control/motion_node.py`）
   - 抓取-放置状态机。使用 `Step` 枚举定义 ROTATE、MOVE_XY、LOWER_Z、GRIP_CLOSE、LIFT_Z、MOVE_PLACE、GRIP_OPEN、RETURN_HOME 等状态。
   - 通过 10 Hz 的 `_control_loop` 定时器推进状态，不再在回调中 `time.sleep` 阻塞 executor。
   - 每步目标位姿、安全高度、home/放置位姿、容差、超时、夹爪等待时间、类别 Z 补偿均从 `src/robot_arm_control/config/motion_config.yaml` 加载。
   - 运行周期中设置 `busy` 标志，忽略新的 `grasp_result`，避免目标被覆盖。
   - 单步超时后发布失败状态并返回 home。

4. **`llm_voice/llm_node`**（可选）
   - 基于智谱 AI 提供 ROS 服务 `/llm/ask_text` 和 `/llm/ask_audio`。
   - `LLMProcessor` 使用 `queue.Queue` + `Future` 实现线程安全，避免并发 service 调用相互覆盖。

### 消息定义

- `robot_arm_interfaces/msg/GraspResult` —— 相机坐标系平移/旋转、抓取宽度/置信度、基坐标系位置/欧拉角、类别名、时间戳。
- `robot_arm_interfaces/msg/RobotInfo` —— 消息头、关节位置、末端位姿、状态字符串、故障标志。

### 视觉与抓取检测（`src/grasp_pipeline`）

`src/grasp_pipeline` 是项目定制的视觉/抓取检测流水线，已按职责拆分为多个工具类：

- `config.Config` 从 `src/vision_grasp/config/grasp_config.yaml` 加载运行时配置，不再硬编码绝对路径。模型权重默认指向 `{ROBOARM_WS}/assets/`。
- `core.model_manager.ModelManager.load_all()` 加载 YOLO（`assets/all.pt`）、SAM（`assets/sam_vit_b_01ec64.pth`）和 GraspNet（`assets/checkpoint.tar`）。
- `core.object_segmentor.ObjectSegmentor.generate_masks_auto()` 运行 YOLO 检测 + SAM 分割，返回掩码路径和检测到的类别名。
- `core.grasp_predictor.GraspPredictor.predict_auto()` 构建带掩码的点云，执行 GraspNet 推理、碰撞检测、NMS，并按接近竖直角度与目标中心距离综合评分，然后通过 `transforms.CoordinateTransformer.grasp_to_base()` 将最优抓取转换到基坐标系。
- `transforms.coordinate_transformer.CoordinateTransformer.grasp_to_base()` 的变换链：抓取→相机对齐 → 相机→末端（手眼） → 末端→基座，最后沿 Z 轴应用 `GRIPPER_LENGTH` 偏移。

底层依赖 `tools/graspnet_baseline` 中的 GraspNet 网络、CUDA 算子与评估库。

### 夹爪控制（`src/robot_arm_control/robot_arm_control`）

夹爪控制代码已内联到 ROS 包中：

- `gripper_can.py` —— 达妙电机 CAN/串口协议（`Motor`、`MotorControl`、帧编解码等）。
- `gripper_driver.py` —— 运行时 `io_node.py` 实际使用的夹爪脚本，暴露 `init_gripper(port)`、`open_gripper()`、`close_gripper()`。

## 重要的硬编码配置

以下默认值仍存在，但已通过 ROS 参数或 YAML 配置文件暴露，迁移时优先修改配置而非源码：

- `src/vision_grasp/vision_grasp/grasp_node.py`：通过 `ROBOARM_WS` 解析 `src/` 根目录。
- `src/robot_arm_control/robot_arm_control/io_node.py`：默认服务器 IP、允许客户端 IP、夹爪串口均可通过 ROS 参数覆盖。夹爪驱动已内联到同包 `gripper_driver.py`，不再通过 `motor_sdk_path` 参数指定外部路径。
- `src/robot_arm_control/robot_arm_control/motion_node.py`：默认读取 `src/robot_arm_control/config/motion_config.yaml`，可在节点参数 `motion_config_path` 中修改。
- `src/llm_voice/llm_voice/llm_node.py`：默认读取 `config/api_keys.yaml`，可在节点参数 `api_key_path` 中修改。
- `src/grasp_pipeline/config.py`：所有模型路径、相机内参、手眼参数、当前末端位姿均从 `src/vision_grasp/config/grasp_config.yaml` 加载。
- `tools/eyeInHand/eye_in_hand.py`：图像目录、棋盘格参数、相机内参。

配置文件清单：

- `config/api_keys.yaml` —— 智谱 API key（已 gitignore，使用 `config/api_keys.yaml.example` 作为模板）。
- `src/robot_arm_control/config/motion_config.yaml` —— 抓取-放置位姿、容差、超时、类别补偿。
- `src/vision_grasp/config/grasp_config.yaml` —— 模型路径、相机内参、手眼标定参数。

## 常见开发注意事项

- 仓库中英文名称和注释混用，新增代码请与所在文件保持一致。
- `grasp_node` 和 `motion_node` 同时包含交互式与自动两种变体；运行时实际使用自动变体（`generate_masks_auto`、`run_grasp_prediction_auto`）。
- ROS 功能包使用 `ament_python` 或 `ament_cmake`。测试包括标准的 `ament_copyright`、`ament_flake8`、`ament_pep257` 和 `pytest`。
- 新增模型权重请放入 `assets/`，不要提交到 Git。
- 新增非 ROS 工具/脚本请放入 `tools/`，不要直接放进 `src/`。

## Git 提交规范

Angular 格式，中文消息：`<type>(<scope>): <简述>`。type: feat/fix/docs/style/refactor/test/chore/perf。
**禁止在 commit message 中添加 `Co-Authored-By: Claude` 等 Claude 协作者信息。**