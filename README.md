# 🤖 机械臂上位机程序

📋 项目简介

本项目是为**机械臂比赛**开发的综合性程序集合，涵盖了**机械臂控制**、**视觉识别**、**抓取点检测**等多个核心模块。作为**机器人学课程比赛作品**，它展示了现代机器人技术的集成应用。

---

## 🌟 项目亮点

🤖 智能交互: 集成大语言模型，实现自然语言控制              
👁️ 精准视觉: 基于深度学习的物体识别与抓取检测                 
🔧 模块化设计: 清晰的代码结构，易于扩展和维护                  
🎯 实时性能: 优化的算法确保实时响应                        

## 🏗️ 项目架构

```text
RoboArm-Vision/                 # ROS 2 workspace 根目录
├── src/                        # 仅放 ROS 2 功能包
│   ├── arm_interfaces/         # 自定义消息/服务接口
│   ├── arm_utils/              # 跨包通用工具函数
│   ├── arm_control/            # 机械臂通信 + 运动状态机
│   ├── vision_grasp/           # RealSense 视觉 + 抓取检测
│   ├── grasp_pipeline/         # YOLO + SAM + GraspNet 抓取检测流水线
│   ├── llm_voice/              # 大语言模型语音交互
│   └── arm_bringup/            # 启动文件
├── assets/                     # 模型权重（大文件不提交到 Git，all.pt 除外）
├── tools/                      # 非 ROS 工具/库
├── config/                     # 共享运行时配置
├── scripts/                    # 常用快捷脚本（标定、抓取、构建、检查等）
├── docs/                       # 详细文档
├── requirements.txt            # Python 依赖
├── README.md
└── CLAUDE.md
```

---

## 🚀 核心功能

| 功能模块                  | 描述                                                         | 技术栈                |
| ------------------------- | ------------------------------------------------------------ | --------------------- |
| 🔧**机械臂控制**    | 通过 Gloria-M SDK 与下位机 TCP 实现精确的机械臂运动控制      | Python, ROS 2         |
| 👁️**视觉识别**    | 利用 Intel RealSense 相机进行高精度图像采集和处理            | OpenCV, RealSense SDK |
| 📐**手眼标定**      | 实现相机坐标系与机械臂坐标系的精确转换                       | 计算机视觉算法        |
| 🎯**抓取点检测**    | 基于 YOLO + SAM + GraspNet 智能检测物体的最佳抓取位置        | PyTorch, CNN          |
| 🗺️**运动规划**    | 抓取-放置状态机，确保安全可靠的抓取任务执行                  | 运动学算法            |
| 🗣️**LLM语音交互** | 通过大语言模型实现语音交互，支持机械臂控制指令和物品信息查询 | ZhipuAI API, 语音识别 |

---

## 🚀 快速开始

1. **安装依赖**

   ```bash
   sudo apt install ros-humble-realsense2-camera ros-humble-cv-bridge
   conda create -n grasp python=3.10
   conda activate grasp
   pip install -r requirements.txt
   ```

2. **下载模型权重**

   ```bash
   ./scripts/download_assets.sh
   ```

3. **配置 API Key**

   ```bash
   cp config/api_keys.yaml.example config/api_keys.yaml
   # 编辑 config/api_keys.yaml 填入智谱 AI 的 API key
   ```

4. **构建工作空间**

   ```bash
   ./scripts/build.sh
   ```

5. **启动系统**

   ```bash
   ros2 launch arm_bringup robot_arm.launch.py
   ```

详细步骤、参数调优与故障排查见 [docs/usage.md](docs/usage.md)。

## 📚 文档

- **[系统设计文档](docs/architecture.md)**：架构、节点关系、数据流、坐标变换、状态机
- **[使用说明](docs/usage.md)**：安装、配置、启动、调试、常见问题

详细内容请查看上述文档。

### 环境要求

- Ubuntu 22.04 / ROS 2 Humble
- Python 3.10+
- CUDA 环境（推荐）
- 安装 ROS 2 依赖：

```bash
sudo apt install ros-humble-realsense2-camera ros-humble-cv-bridge
```

- 安装 Python 依赖：

```bash
conda activate grasp
pip install -r requirements.txt
```

### 配置环境变量

`scripts/` 下的快捷脚本会根据自身位置自动推断仓库根目录，通常无需设置环境变量。如需强制指定，可在 `~/.bashrc` 中设置：

```bash
export ROBOARM_WS=/path/to/RoboArm-Vision
```

### 构建工作空间

```bash
cd $ROBOARM_WS
./scripts/build.sh
```

或手动：

```bash
cd $ROBOARM_WS
conda activate grasp
python -m colcon build --symlink-install
source install/setup.bash
```

### 配置 API Key

LLM 语音/文本节点需要智谱 AI 的 API key：

```bash
cp config/api_keys.yaml.example config/api_keys.yaml
# 编辑 config/api_keys.yaml，填入你的 zhipu_api_key
```

`config/api_keys.yaml` 已加入 `.gitignore`，不会被提交到 Git。

### 启动系统

一键启动三个核心节点：

```bash
ros2 launch arm_bringup robot_arm.launch.py
```

或分别启动：

```bash
# 终端 1：视觉 / 抓取检测
ros2 run vision_grasp grasp_node

# 终端 2：上位机通信 + 夹爪控制
ros2 run arm_control io_node

# 终端 3：运动状态机
ros2 run arm_control motion_node

# 终端 4（可选）：大语言模型交互
ros2 run llm_voice llm_node
```

### 模型权重

预训练权重统一放到 `assets/` 目录下：

- `assets/all.pt` —— YOLO 检测 / 分割模型（**已包含在仓库中**）
- `assets/sam_vit_b_01ec64.pth` —— SAM 分割模型（可选）
- `assets/checkpoint.tar` —— GraspNet 抓取模型

缺失的权重可一键下载：

```bash
./scripts/download_assets.sh
```

> 注意：`assets/` 下的大模型文件已加入 `.gitignore`，不会提交到 Git；`all.pt` 除外。新克隆仓库后请运行下载脚本或按 `assets/README.md` 手动获取。

---

## 🔌 接口与服务

主要 ROS 2 话题与服务：

| 名称 | 类型 | 说明 |
|------|------|------|
| `/grasp_result` | `arm_interfaces/msg/GraspResult` | 视觉节点发布的最佳抓取位姿 |
| `/robot_status` | `std_msgs/msg/String` | 运动节点状态反馈，如 `have backed` |
| `/RobotInfo` | `arm_interfaces/msg/RobotInfo` | 下位机上传的机械臂位姿 |
| `/RobotMove` | `trajectory_msgs/msg/JointTrajectory` | 运动指令（下位机接收） |
| `/GripperControl` | `std_msgs/msg/String` | 夹爪控制：`open` / `close` |
| `/llm/ask_text` | `llm_voice/srv/AskText` | 文本问答服务 |
| `/llm/ask_audio` | `std_srvs/srv/Trigger` | 语音问答服务 |

完整消息/服务定义见 `src/arm_interfaces/` 与 `src/llm_voice/srv/`。

## 📐 手眼标定

使用快捷脚本：

```bash
cd $ROBOARM_WS
./scripts/calib_capture.sh   # 采集约 30 张棋盘格图像
./scripts/calib_run.sh       # 计算相机到机械臂末端的变换
```

或手动进入 `tools/eyeInHand` 目录运行：

```bash
cd tools/eyeInHand
python3 保存RGB图像.py   # 采集约 30 张棋盘格图像
python3 eye_in_hand.py   # 计算相机到机械臂末端的变换
```

---

## 📅 开发历程

### 🎯 **2025年开发里程碑**

| 日期               | 进展                       | 状态    |
| ------------------ | -------------------------- | ------- |
| **12月19日** | 🎨 优化运行效果            | ✅ 完成 |
| **12月18日** | 🔗 完成联调测试            | ✅ 完成 |
| **12月17日** | 🧠 封装大语言模型功能      | ✅ 完成 |
| **12月16日** | 🤖 完成机械臂控制部分测试  | ✅ 完成 |
| **12月15日** | 📡 完成上位机与控制端通信  | ✅ 完成 |
| **12月12日** | 🔧 完成通信自测 & 电控测试 | ✅ 完成 |
| **12月10日** | 📐 完成手眼标定            | ✅ 完成 |
| **12月5日**  | ⚙️ 完成手眼标定配置      | ✅ 完成 |
| **12月3日**  | 🚀 完成ORIN NX移植         | ✅ 完成 |

---

## 🛠️ 技术栈

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/C++-00599C?style=flat&logo=c%2B%2B&logoColor=white" alt="C++">
  <img src="https://img.shields.io/badge/ROS-22314E?style=flat&logo=ros&logoColor=white" alt="ROS">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA">
    <img src="https://img.shields.io/badge/ZhipuAI-API-orange?style=flat&logo=ai&logoColor=white" alt="ZhipuAI">
</div>

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

- 发现 bug 或有新需求，请先搜索现有 Issue，避免重复。
- 提交 PR 前请确保代码可通过 `pytest` 与 `flake8` 基础检查。
- 提交信息使用 Angular 格式，中文简述，例如：
  `feat(scripts): 添加新的调试启动脚本`。

详细开发约定见 [CLAUDE.md](CLAUDE.md)。

---

## 📄 许可证

本项目采用 [Apache-2.0](LICENSE) 许可证。使用第三方模型权重（如 GraspNet、SAM、YOLO）时，请遵守各权重自身的许可协议。

---

<div align="center">
  ⭐ 如果这个项目对你有帮助，请给个 Star！
</div>
