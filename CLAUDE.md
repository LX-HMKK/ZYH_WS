# CLAUDE.md

本文件为 Claude Code（claude.ai/code）提供在本仓库中工作时的快速指引。详细设计、安装、配置、调试说明请查阅 `docs/`。

## 项目概述

本项目是 2025 年机器人学课程的比赛作品——“机械臂上位机程序”。它集成了六自由度机械臂、Intel RealSense 视觉、基于深度学习的抓取检测以及大语言模型语音交互。

本仓库现在按标准 ROS 2 workspace 组织：

- `src/` —— 仅包含 ROS 2 功能包。
- `assets/` —— 模型权重等大文件（已加入 `.gitignore`，不提交到 Git）。
- `tools/` —— 非 ROS 的工具/库/脚本。
- `config/` —— 共享运行时配置（API key 等）。
- `docs/` —— 详细设计文档与使用说明。

主要子系统：

- `src/arm_bringup/` —— 启动文件（launch）。
- `src/arm_control/` —— 下位机 TCP 通信、夹爪控制、抓取-放置状态机。
- `src/vision_grasp/` —— RealSense 视觉节点，负责调用 `grasp_pipeline` 并发布抓取结果。
- `src/grasp_pipeline/` —— YOLO + SAM + GraspNet 抓取检测流水线及工具类。
- `src/llm_voice/` —— 智谱 AI 大语言模型语音/文本交互。
- `src/arm_interfaces/` —— 自定义消息/服务定义。
- `src/arm_utils/` —— 跨包通用工具（路径解析、配置加载、QoS、验证等）。
- `tools/graspnet_baseline/` —— GraspNet 基线网络、原生算子与评估库。
- `tools/eyeInHand/` —— 基于棋盘格的手眼标定。

## 参考文档

- `docs/architecture.md` —— 系统设计文档（架构、数据流、坐标变换、状态机、TCP/夹爪/LLM 细节）。
- `docs/usage.md` —— 使用说明（安装、配置、启动、调试、常见问题、硬编码配置清单）。
- `README.md` —— 项目简介与快速开始。

## 路径约定

所有 ROS 节点通过 `ROBOARM_WS` 环境变量定位仓库根目录，未设置时默认回退到 `/home/zyh/ZYH_WS`。

```python
def get_workspace_root() -> str:
    return os.environ.get("ROBOARM_WS", "/home/zyh/ZYH_WS")
```

典型路径：

- 模型权重：`{ROBOARM_WS}/assets/all.pt`、`{ROBOARM_WS}/assets/sam_vit_b_01ec64.pth`
- 视觉/抓取配置：`{ROBOARM_WS}/src/grasp_pipeline/config/grasp_config.yaml`
- 运动配置：`{ROBOARM_WS}/src/arm_control/config/motion_config.yaml`
- API key：`{ROBOARM_WS}/config/api_keys.yaml`

## 构建与运行

```bash
export ROBOARM_WS=/home/zyh/ZYH_WS
cd $ROBOARM_WS
conda activate grasp
python -m colcon build --symlink-install
source install/setup.bash
```

一键启动：

```bash
ros2 launch arm_bringup robot_arm.launch.py
```

或分别启动：

```bash
ros2 run vision_grasp grasp_node          # 视觉 / 抓取检测
ros2 run arm_control io_node        # TCP 通信 + 夹爪
ros2 run arm_control motion_node    # 运动状态机
ros2 run llm_voice llm_node               # LLM 语音/文本交互（可选）
```

`llm_node` 默认读取 `config/api_keys.yaml`，可通过 ROS 参数覆盖：

```bash
ros2 run llm_voice llm_node --ros-args -p api_key_path:=/path/to/api_keys.yaml
```

## 无硬件时测试抓取流程

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

## 常见开发注意事项

- 仓库中英文名称和注释混用，新增代码请与所在文件保持一致。
- `grasp_node` 和 `motion_node` 同时包含交互式与自动两种变体；运行时实际使用自动变体（`generate_masks_auto`、`run_grasp_prediction_auto`）。
- ROS 功能包使用 `ament_python` 或 `ament_cmake`。测试包括标准的 `ament_copyright`、`ament_flake8`、`ament_pep257` 和 `pytest`。
- 新增模型权重请放入 `assets/`，不要提交到 Git。
- 新增非 ROS 工具/脚本请放入 `tools/`，不要直接放进 `src/`。
- 新增详细说明请放入 `docs/`，并在 `README.md` / `CLAUDE.md` 中引用。

## Git 提交规范

Angular 格式，中文消息：`<type>(<scope>): <简述>`。type: feat/fix/docs/style/refactor/test/chore/perf。
**禁止在 commit message 中添加 `Co-Authored-By: Claude` 等 Claude 协作者信息。**
