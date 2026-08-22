# 机械臂上位机系统设计文档

本文档描述 RoboArm-Vision 项目的整体架构、核心模块、数据流与坐标变换。

## 1. 系统概述

RoboArm-Vision 是一套面向机械臂抓取比赛的 ROS 2 上位机系统，集成：

- **视觉感知**：Intel RealSense D435/D435i 深度相机
- **目标检测与分割**：YOLOv8 + Segment Anything Model (SAM)
- **抓取位姿预测**：GraspNet 基线网络
- **机械臂控制**：下位机 TCP 通信、夹爪控制、抓取-放置状态机
- **人机交互**：智谱 AI 大语言模型语音/文本交互

系统按标准 ROS 2 workspace 组织，所有节点通过 `ROBOARM_WS` 环境变量定位仓库根目录。

## 2. 软件架构

### 2.1 包结构

```text
RoboArm-Vision/                 # ROS 2 workspace 根目录
├── src/                        # ROS 2 功能包
│   ├── arm_bringup/      # 启动文件
│   ├── arm_interfaces/   # 自定义消息/服务接口
│   ├── arm_utils/        # 跨包通用工具函数
│   ├── arm_control/      # 机械臂通信 + 夹爪 + 运动状态机
│   ├── vision_grasp/           # RealSense 视觉 + 抓取检测节点
│   ├── grasp_pipeline/         # YOLO + SAM + GraspNet 抓取检测库
│   └── llm_voice/              # 大语言模型语音交互
├── assets/                     # 模型权重（大文件，不提交到 Git）
├── tools/                      # 非 ROS 工具/库
│   ├── eyeInHand/              # 手眼标定
│   └── graspnet_baseline/      # GraspNet 基线网络、算子、API
├── config/                     # 共享运行时配置（API key 等）
└── docs/                       # 设计文档与使用说明
```

### 2.1.1 `arm_utils`

为避免各功能包重复定义相同工具函数，统一放在 `arm_utils`：

| 模块 | 文件 | 职责 |
|------|------|------|
| `get_workspace_root` | `path_utils.py` | 通过 `ROBOARM_WS` 解析仓库根目录 |
| `load_yaml_config` / `load_zhipu_api_key` | `config_utils.py` | YAML 配置、API key 加载 |
| `is_float` | `validation_utils.py` | 字符串 float 校验 |
| `reliable_qos` | `qos_utils.py` | 可靠的 ROS 2 QoSProfile |

### 2.2 运行时节点

```text
┌─────────────────┐     /grasp_result      ┌──────────────────┐
│  vision_grasp   │ ─────────────────────→ │ arm_control│
│   grasp_node    │   (GraspResult)        │   motion_node    │
└─────────────────┘                        └────────┬─────────┘
        ↑                                           │
        │                                           │ RobotMove
        │                                           │ GripperControl
        │                                           ↓
        │                                    ┌──────────────────┐
        │                                    │ arm_control│
        └────────────────────────────────────│    io_node       │
              /robot_status (String)         │  (TCP + gripper) │
                                             └──────────────────┘
                                                      │
                                                      │ TCP
                                                      ↓
                                               ┌──────────────┐
                                               │   下位机      │
                                               │  机械臂控制器 │
                                               └──────────────┘

可选节点：
┌─────────────────┐
│    llm_voice    │ 提供 /llm/ask_text, /llm/ask_audio 服务
│    llm_node     │
└─────────────────┘
```

| 节点 | 包 | 职责 |
|------|-----|------|
| `vision_grasp/grasp_node` | `vision_grasp` | 采集 RealSense 帧，YOLO 检测 → SAM 分割 → GraspNet 抓取预测 → 坐标转换 → 发布 `/grasp_result` |
| `arm_control/io_node` | `arm_control` | TCP 服务器接收下位机状态，发布 `/RobotInfo`；订阅 `RobotMove` 转发给下位机；订阅 `GripperControl` 控制夹爪 |
| `arm_control/motion_node` | `arm_control` | 抓取-放置状态机：订阅 `/grasp_result` 和 `/RobotInfo`，发布 `RobotMove`、`GripperControl`、`robot_status` |
| `llm_voice/llm_node` | `llm_voice` | 提供文本/语音交互服务，可查询物品信息或接收自然语言指令 |

### 2.3 自定义接口

定义在 `arm_interfaces`：

- **`GraspResult`** (`arm_interfaces/msg/GraspResult`)
  - 相机坐标系抓取位姿 (`trans_cam`, `rot_cam_flat`)
  - 基坐标系目标位姿 (`pos_base`, `euler_base`)
  - 抓取宽度 `width`、置信度 `score`
  - 类别名 `cls_name`、时间戳 `stamp`

- **`RobotInfo`** (`arm_interfaces/msg/RobotInfo`)
  - 消息头 `header`
  - 关节位置 `joint_positions`
  - 末端位姿 `end_positions`
  - 状态字符串 `state`
  - 故障标志 `fault_flag`

### 2.4 数据流

一次完整的抓取周期：

1. `vision_grasp/grasp_node` 持续采集图像，检测并发布 `/grasp_result`。
2. `arm_control/motion_node` 收到 `/grasp_result` 后启动状态机。
3. 状态机按 `ROTATE_TO_PICK → LIFT_TO_PICK_SAFE → MOVE_TO_PICK_XY → LOWER_TO_PICK → GRIP_CLOSE → LIFT_TO_PLACE_SAFE → ROTATE_TO_PLACE → MOVE_TO_PLACE_XY → LOWER_TO_PLACE → GRIP_OPEN → LIFT_TO_HOME_SAFE → RETURN_HOME` 推进，其中 `RETURN_HOME` 进一步拆分为 `ROTATE_TO_HOME → MOVE_TO_HOME_XY → LOWER_TO_HOME`。
4. 每个运动步骤通过 `RobotMove`（`JointTrajectory`）发送给 `arm_control/io_node`。
5. `io_node` 通过 TCP 将目标位姿转发给下位机机械臂控制器。
6. 下位机实时返回当前位姿，`io_node` 解析后发布 `/RobotInfo`。
7. `motion_node` 比较当前位姿与目标位姿，到达容差后进入下一步。
8. 夹爪开合通过 `GripperControl`（`std_msgs/String`）由 `io_node` 转发给夹爪驱动。
9. 周期完成后 `motion_node` 发布 `robot_status = "have backed"`。
10. `grasp_node` 收到 `"have backed"` 后触发下一次检测。

## 3. 抓取检测流水线

`grasp_pipeline` 是项目定制的视觉/抓取检测库，被 `vision_grasp/grasp_node` 调用。

### 3.1 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| `Config` | `grasp_pipeline/config.py` | 从 `src/grasp_pipeline/config/grasp_config.yaml` 加载运行时配置 |
| `ModelManager` | `grasp_pipeline/core/model_manager.py` | 加载 YOLO、SAM、GraspNet 模型 |
| `ObjectSegmentor` | `grasp_pipeline/core/object_segmentor.py` | YOLO 检测 + SAM 分割，生成目标掩码 |
| `DataProcessor` | `grasp_pipeline/core/data_processor.py` | 点云预处理、采样、碰撞检测 |
| `GraspPredictor` | `grasp_pipeline/core/grasp_predictor.py` | GraspNet 推理、NMS、评分、坐标转换 |
| `CoordinateTransformer` | `grasp_pipeline/transforms/coordinate_transformer.py` | 相机坐标系 → 机械臂基坐标系 |
| `vision_utils` | `grasp_pipeline/utils/vision_utils.py` | IoU、NMS 等视觉辅助函数 |

RealSense 相机驱动与帧预处理已拆分到 `vision_grasp`：

| 模块 | 文件 | 职责 |
|------|------|------|
| `FrameProcessor` | `vision_grasp/frame_processor.py` | RealSense 帧对齐与预处理 |
| `RealSenseCamera` | `vision_grasp/camera_driver.py` | RealSense 初始化、重启、获取对齐帧、资源释放 |

### 3.2 坐标变换链

`CoordinateTransformer.grasp_to_base()` 执行以下变换：

```text
抓取位姿（GraspNet 相机坐标系）
        ↓  应用坐标系对齐矩阵 r_adjust
相机对齐坐标系
        ↓  手眼标定 T_cam→end
末端法兰坐标系
        ↓  当前末端位姿 T_end→base
机械臂基座坐标系
        ↓  沿 Z 轴补偿 GRIPPER_LENGTH
最终目标位姿（基座标系）
```

关键参数：

- `current_ee_pose`：机械臂当前末端位姿 `[x, y, z, rx, ry, rz]`（基座标系，单位 m / rad）
- `handeye_rot` / `handeye_trans`：相机相对于末端的旋转和平移
- `gripper_length`：夹爪长度补偿（沿末端 Z 轴负方向）

## 4. 运动状态机

抓取-放置状态机实现位于 `arm_control/arm_control/motion_state_machine.py` 中的 `PickPlaceStateMachine`，`arm_control/motion_node` 是其 ROS 2 包装节点，负责订阅 `/grasp_result`、`/RobotInfo` 并发布 `RobotMove`、`GripperControl`、`robot_status`。

### 4.1 状态定义

完整抓取-放置周期按以下顺序推进：

```text
ROTATE_TO_PICK → LIFT_TO_PICK_SAFE → MOVE_TO_PICK_XY → LOWER_TO_PICK → GRIP_CLOSE
→ LIFT_TO_PLACE_SAFE → ROTATE_TO_PLACE → MOVE_TO_PLACE_XY → LOWER_TO_PLACE → GRIP_OPEN
→ LIFT_TO_HOME_SAFE → ROTATE_TO_HOME → MOVE_TO_HOME_XY → LOWER_TO_HOME
```

| 状态 | 说明 |
|------|------|
| `IDLE` | 空闲，等待 `/grasp_result` |
| `ROTATE_TO_PICK` | 在当前高度旋转到目标拾取偏航角 |
| `LIFT_TO_PICK_SAFE` | 抬升 Z 轴到安全高度，保持拾取姿态 |
| `MOVE_TO_PICK_XY` | 在安全高度移动到拾取 XY 位置 |
| `LOWER_TO_PICK` | Z 轴下降到抓取高度，应用类别补偿 |
| `GRIP_CLOSE` | 闭合夹爪并等待 |
| `LIFT_TO_PLACE_SAFE` | 从拾取点抬升 Z 轴到安全高度 |
| `ROTATE_TO_PLACE` | 在安全高度旋转到放置姿态 |
| `MOVE_TO_PLACE_XY` | 在安全高度移动到放置 XY 位置 |
| `LOWER_TO_PLACE` | Z 轴下降到放置高度 |
| `GRIP_OPEN` | 打开夹爪并等待 |
| `LIFT_TO_HOME_SAFE` | 从放置点抬升 Z 轴到安全高度 |
| `ROTATE_TO_HOME` | 在安全高度旋转到 home 姿态 |
| `MOVE_TO_HOME_XY` | 在安全高度移动到 home XY 位置 |
| `LOWER_TO_HOME` | Z 轴下降到 home 高度 |

### 4.1.1 笛卡尔安全约束

为避免机械臂在运动中与场景或自身发生碰撞，状态机将任意两点间的运动拆分为严格的四步序列：

1. **先旋转**：姿态调整优先在起点 Z 高度（拾取阶段）或安全高度（放置/归位阶段）完成，避免在低位旋转时夹爪扫过障碍物。
2. **再抬升 Z**：旋转完成后抬升到 `safe_z_height` 高空，确保后续水平移动时夹爪高于物体和台面。
3. **XY 平面移动**：仅在安全高度进行水平移动，禁止在低位做大范围平移。
4. **最后下降 Z**：到达目标正上方后再下降 Z 轴到工作高度（抓取、放置或 home）。

中止周期或失败时同样遵循该安全约束返回 home。

### 4.2 位置到达判定

- 平移误差 `< position_tolerance`（默认 5 mm）
- 旋转误差 `< rotation_tolerance`（默认 0.5 deg）
- `ROTATE` 步骤仅检查旋转误差

### 4.3 超时与安全

- 单步超时 `step_timeout`（默认 10 s），超时后中止周期并返回 home
- 运行周期中设置 `busy` 标志，忽略新的 `/grasp_result`，避免目标被覆盖

## 5. TCP 通信协议

`arm_control/io_node` 作为 TCP 服务器监听下位机连接。

- 默认服务器地址：`172.16.26.125:10001`
- 默认允许客户端：`172.16.26.126`
- 向下位机发送：`[x,y,z,rx,ry,rz]`（ASCII，单位与下位机约定一致）
- 接收下位机数据：解析 `get #real#6#x,y,z,rx,ry,rz` 格式并发布 `RobotInfo`

以上参数均可通过 ROS 参数覆盖。

## 6. 夹爪控制

夹爪采用达妙电机 CAN/串口协议，封装在 `arm_control/arm_control/gripper_can.py` 和 `gripper_driver.py` 中。

- `GripperController(serial_port)`：初始化串口并使能电机
- `GripperController.open_gripper()`：打开夹爪
- `GripperController.close_gripper()`：闭合夹爪

`io_node` 中的 `GripperROSAdapter` 持有 `GripperController` 实例，通过 `GripperControl` 话题接收 `"open"` / `"close"` 指令并调用对应方法。

## 7. 大语言模型交互

`llm_voice/llm_node` 基于智谱 AI 提供 ROS 服务：

- `/llm/ask_text` (`arm_interfaces/srv/AskText`)：文本问答
- `/llm/ask_audio` (`std_srvs/srv/Trigger`)：语音问答（Linux only）

启动前需要配置 `config/api_keys.yaml` 中的 `zhipu_api_key`。

## 8. 配置文件

| 配置文件 | 说明 |
|----------|------|
| `src/grasp_pipeline/config/grasp_config.yaml` | 模型路径、相机内参、手眼标定、GraspNet 参数 |
| `src/arm_control/config/motion_config.yaml` | home/放置位姿、容差、超时、类别 Z 补偿 |
| `config/api_keys.yaml` | 智谱 API key（gitignore，需手动创建） |

详见 [使用说明](./usage.md)。

## 9. 坐标系约定

- **相机坐标系**：RealSense 默认坐标系，Z 轴指向场景前方
- **末端法兰坐标系**：机械臂末端法兰坐标系
- **机械臂基座坐标系**：机械臂底座坐标系
- 手眼标定结果描述的是 **相机相对于末端** 的位姿

具体标定方法见 `tools/eyeInHand/`。

## 10. 扩展建议

- 新增物体类别：修改 `src/grasp_pipeline/config/grasp_config.yaml` 中的 `class_compensation` 和运动配置中的 `class_compensation`
- 更换相机：更新 `depth_intr` 内参和 `camera_res`
- 调整抓取策略：修改 `grasp_pipeline/core/grasp_predictor.py` 中的评分权重
- 接入其他 LLM：修改 `llm_voice/llm_voice/llm_module.py` 中的 `_chat_with_glm`
