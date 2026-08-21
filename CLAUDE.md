# CLAUDE.md

本文件为 Claude Code（claude.ai/code）提供在本仓库中工作时的指导。

## 项目概述

本项目是 2025 年机器人学课程的比赛作品——“机械臂上位机程序”。它集成了六自由度机械臂、Intel RealSense 视觉、基于深度学习的抓取检测以及大语言模型语音交互。

本仓库现在按标准 ROS 2 workspace 组织：

- `src/` —— 仅包含 ROS 2 功能包。
- `assets/` —— 模型权重等大文件（已加入 `.gitignore`，不提交到 Git）。
- `tools/` —— 非 ROS 的工具/库/脚本。
- `config/` —— 共享运行时配置（API key 等）。

主要子系统：

- `src/robot_arm_bringup/` —— 启动文件（launch）。
- `src/codroid_node/` —— 下位机 TCP 通信、夹爪控制、抓取-放置状态机。
- `src/grasp_publisher/` —— RealSense 视觉、YOLO + SAM + GraspNet 抓取检测。
- `src/llm_voice/` —— 智谱 AI 大语言模型语音/文本交互。
- `src/codroid_msgs/`、`src/grasp_interfaces/` —— 自定义消息/服务定义。
- `tools/graspnet-baseline-main/` —— GraspNet 基线网络及项目定制集成代码 `kw/robot.py`。
- `tools/Gloria-M-SDK-1.0.0/` —— Gloria-M 夹爪串口 SDK。
- `tools/eyeInHand/` —— 基于棋盘格的手眼标定。
- `tools/librealsense-master/` —— RealSense D435 辅助脚本。
- `tools/motor/` —— 额外的电机接口。
- `tools/label_process/` —— 标签处理工具。

## 路径约定

所有 ROS 节点通过 `ROBOARM_WS` 环境变量定位仓库根目录，未设置时默认回退到 `/home/zyh/ZYH_WS`。

```python
def get_workspace_root() -> str:
    return os.environ.get("ROBOARM_WS", "/home/zyh/ZYH_WS")
```

典型路径：

- 模型权重：`{ROBOARM_WS}/assets/all.pt`、`{ROBOARM_WS}/assets/sam_vit_b_01ec64.pth`
- GraspNet 工具根目录：`{ROBOARM_WS}/tools/graspnet-baseline-main`
- 电机 SDK：`{ROBOARM_WS}/tools/Gloria-M-SDK-1.0.0/motor`
- API key：`{ROBOARM_WS}/config/api_keys.yaml`
- 运动配置：`{ROBOARM_WS}/src/codroid_node/config/motion_config.yaml`
- 抓取配置：`{ROBOARM_WS}/src/grasp_publisher/config/grasp_config.yaml`

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
ros2 run grasp_publisher grasp_node      # 视觉 / 抓取检测
ros2 run codroid_node codroid_io         # 上位机与下位机 TCP 通信 + 夹爪控制
ros2 run codroid_node codroid_move_test  # 运动状态机
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
ros2 topic pub /grasp_result grasp_interfaces/msg/GraspResult "{
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

### Gloria-M SDK（`tools/Gloria-M-SDK-1.0.0`）

以可编辑模式安装：

```bash
cd tools/Gloria-M-SDK-1.0.0
pip install -r requirements.txt
pip install -e .
```

运行单元测试：

```bash
pytest -q
```

运行示例：

```bash
python examples/basic_usage.py [PORT]   # 交互式；也支持环境变量 GLORIA_PORT
python examples/advanced_usage.py       # 模拟 ACK，无需硬件
```

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

### GraspNet 基线（`tools/graspnet-baseline-main`）

安装原生算子（首次运行前必需）：

```bash
cd tools/graspnet-baseline-main
pip install -r requirements.txt
cd pointnet2 && python setup.py install
cd ../knn && python setup.py install
cd ../graspnetAPI-master && pip install .
```

运行交互式抓取检测演示（需要模型权重和 RealSense）：

```bash
cd tools/graspnet-baseline-main
python demo.py --checkpoint_path <路径> ...
```

训练/测试命令分别位于 `command_train.sh`、`command_test.sh` 和 `command_demo.sh`。项目定制的自动流程在 `kw/robot.py` 中，由 ROS `grasp_node` 调用。

## 高层架构

### ROS 2 节点关系图

运行时由三个核心节点和一个可选节点组成：

1. **`grasp_publisher/grasp_node`**（`src/grasp_publisher/grasp_publisher/grasp_node.py`）
   - 采集 RealSense 帧，依次运行 YOLOv8 检测、SAM 分割、GraspNet 抓取预测，将最优抓取从相机坐标系转换到机械臂基坐标系，并向 `/grasp_result` 发布 `grasp_interfaces/GraspResult`。
   - 订阅 `/robot_status`，收到 `"have backed"` 后触发下一次检测。
   - 帧超时时会自动重启相机（最多 3 次）。
   - 临时图像使用 `tempfile.NamedTemporaryFile`，不再固定写入 `/tmp`。
   - 通过 `sys.path.append("{ROBOARM_WS}/tools/graspnet-baseline-main")` 导入 `kw.robot`。

2. **`codroid_node/codroid_io`**（`src/codroid_node/codroid_node/codroid_io.py`）
   - 由三个专注的类组成：
     - `RobotTcpServer`：非阻塞 TCP 服务器，监听下位机连接，维护连接池，接收/发送数据。
     - `GripperController`：封装 `tools/Gloria-M-SDK-1.0.0/motor/DM_Motor_Test.py` 的夹爪初始化、打开、闭合，支持失败后定时重连。
     - `CodroidIONode`：ROS 节点，聚合上述两个子系统；订阅 `RobotMove` 并转发给所有下位机；订阅 `GripperControl` 控制夹爪；发布 `RobotInfo`。
   - 服务器 IP、端口、允许客户端 IP、夹爪串口、电机 SDK 路径均通过 ROS 参数暴露，默认值保留原项目配置。

3. **`codroid_node/codroid_move_test`**（`src/codroid_node/codroid_node/codroid_move_test.py`）
   - 抓取-放置状态机。使用 `Step` 枚举定义 ROTATE、MOVE_XY、LOWER_Z、GRIP_CLOSE、LIFT_Z、MOVE_PLACE、GRIP_OPEN、RETURN_HOME 等状态。
   - 通过 10 Hz 的 `_control_loop` 定时器推进状态，不再在回调中 `time.sleep` 阻塞 executor。
   - 每步目标位姿、安全高度、home/放置位姿、容差、超时、夹爪等待时间、类别 Z 补偿均从 `src/codroid_node/config/motion_config.yaml` 加载。
   - 运行周期中设置 `busy` 标志，忽略新的 `grasp_result`，避免目标被覆盖。
   - 单步超时后发布失败状态并返回 home。

4. **`llm_voice/llm_node`**（可选）
   - 基于智谱 AI 提供 ROS 服务 `/llm/ask_text` 和 `/llm/ask_audio`。
   - `LLMProcessor` 使用 `queue.Queue` + `Future` 实现线程安全，避免并发 service 调用相互覆盖。

### 消息定义

- `grasp_interfaces/msg/GraspResult` —— 相机坐标系平移/旋转、抓取宽度/置信度、基坐标系位置/欧拉角、类别名、时间戳。
- `codroid_msgs/msg/RobotInfo` —— 消息头、关节位置、末端位姿、状态字符串、故障标志。

### 视觉与抓取检测（`tools/graspnet-baseline-main/kw/robot.py`）

`kw/robot.py` 是围绕 GraspNet 基线的项目定制集成层：

- `Config` 类现在从 `src/grasp_publisher/config/grasp_config.yaml` 加载，不再硬编码绝对路径。模型权重默认指向 `{ROBOARM_WS}/assets/`。
- `load_all_models()` 加载 YOLO（`assets/all.pt`）、SAM（`assets/sam_vit_b_01ec64.pth`）和 GraspNet（`assets/checkpoint.tar`）。
- `generate_masks_auto()` 运行 YOLO 检测 + SAM 分割，返回掩码路径和检测到的类别名。
- `run_grasp_prediction_auto()` 构建带掩码的点云，执行 GraspNet 推理、碰撞检测、NMS，并按接近竖直角度与目标中心距离综合评分，然后通过 `convert_grasp_to_robot_base()` 将最优抓取转换到基坐标系。
- `convert_grasp_to_robot_base()` 的变换链：抓取→相机对齐 → 相机→末端（手眼） → 末端→基座，最后沿 Z 轴应用 `GRIPPER_LENGTH` 偏移。

### 夹爪控制（`tools/Gloria-M-SDK-1.0.0`）

- `gloria_msdk/` 是一个小型串口 SDK，包括 `SerialManager`、`SerialConfig`、`Command`、`DeviceStatus` 和带 CRC16 的自定义帧协议。
- `motor/DM_Motor_Test.py` 是运行时 `codroid_io.py` 实际使用的夹爪脚本，暴露 `init_gripper(port)`、`open_gripper()`、`close_gripper()`。

## 重要的硬编码配置

以下默认值仍存在，但已通过 ROS 参数或 YAML 配置文件暴露，迁移时优先修改配置而非源码：

- `src/grasp_publisher/grasp_publisher/grasp_node.py`：通过 `ROBOARM_WS` 解析 GraspNet 工具根目录。
- `src/codroid_node/codroid_node/codroid_io.py`：默认电机 SDK 路径、服务器 IP、允许客户端 IP、夹爪串口均可通过 ROS 参数覆盖。
- `src/codroid_node/codroid_node/codroid_move_test.py`：默认读取 `src/codroid_node/config/motion_config.yaml`，可在节点参数 `motion_config_path` 中修改。
- `src/llm_voice/llm_voice/llm_node.py`：默认读取 `config/api_keys.yaml`，可在节点参数 `api_key_path` 中修改。
- `tools/graspnet-baseline-main/kw/robot.py`：所有模型路径、相机内参、手眼参数、当前末端位姿均从 `src/grasp_publisher/config/grasp_config.yaml` 加载。
- `tools/eyeInHand/eye_in_hand.py`：图像目录、棋盘格参数、相机内参。

配置文件清单：

- `config/api_keys.yaml` —— 智谱 API key（已 gitignore，使用 `config/api_keys.yaml.example` 作为模板）。
- `src/codroid_node/config/motion_config.yaml` —— 抓取-放置位姿、容差、超时、类别补偿。
- `src/grasp_publisher/config/grasp_config.yaml` —— 模型路径、相机内参、手眼标定参数。

## 常见开发注意事项

- 仓库中英文名称和注释混用，新增代码请与所在文件保持一致。
- `grasp_node` 和 `codroid_move_test` 同时包含交互式与自动两种变体；运行时实际使用自动变体（`generate_masks_auto`、`run_grasp_prediction_auto`）。
- ROS 功能包使用 `ament_python` 或 `ament_cmake`。测试包括标准的 `ament_copyright`、`ament_flake8`、`ament_pep257` 和 `pytest`。
- 新增模型权重请放入 `assets/`，不要提交到 Git。
- 新增非 ROS 工具/脚本请放入 `tools/`，不要直接放进 `src/`。

## Git 提交规范

Angular 格式，中文消息：`<type>(<scope>): <简述>`。type: feat/fix/docs/style/refactor/test/chore/perf。
**禁止在 commit message 中添加 `Co-Authored-By: Claude` 等 Claude 协作者信息。**