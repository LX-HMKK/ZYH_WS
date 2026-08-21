<p align="center">
  <h1 align="center">🤖 RoboArm-Vision</h1>
  <h3 align="center">机械臂上位机程序</h3>
  <p align="center">
    基于 ROS 2 + RealSense + YOLO / SAM / GraspNet 的智能抓取系统
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white" alt="ROS 2 Humble">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
    <img src="https://img.shields.io/badge/RealSense-0071C5?logo=intel&logoColor=white" alt="RealSense">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  </p>
</p>

<p align="center">
  <a href="#intro">项目简介</a> •
  <a href="#highlights">项目亮点</a> •
  <a href="#architecture">项目架构</a> •
  <a href="#features">核心功能</a> •
  <a href="#quickstart">快速开始</a> •
  <a href="#interfaces">接口与服务</a> •
  <a href="#docs">详细文档</a>
</p>

---

<a name="intro"></a>
## 📋 项目简介

本项目是为**机械臂比赛**开发的综合性上位机程序集合，涵盖**机械臂控制**、**视觉识别**、**抓取点检测**与**大语言模型语音交互**等核心模块。作为机器人学课程比赛作品，它展示了现代机器人技术的集成应用。

> 💡 **目标场景**：在真实比赛环境中，通过 RealSense 相机识别目标物体，利用深度学习模型生成 6D 抓取位姿，并驱动六自由度机械臂完成抓取-放置任务。

---

<a name="highlights"></a>
## 🌟 项目亮点

| 亮点 | 说明 |
|------|------|
| 🤖 **LLM 智能交互** | 集成智谱 AI 大模型，支持自然语言控制与物品信息查询 |
| 👁️ **端到端视觉** | YOLO 检测 → SAM / YOLO-seg 分割 → GraspNet 抓取位姿预测 |
| 🔧 **模块化设计** | ROS 2 功能包拆分清晰，便于扩展与维护 |
| 🎯 **一键抓取** | 提供标定、单次抓取、循环抓取、语音抓取等多种快捷脚本 |
| ⚡ **实时性能** | 针对 RealSense 深度图与点云推理进行优化 |

---

<a name="architecture"></a>
## 🏗️ 项目架构

<details>
<summary>点击展开目录结构</summary>

```text
RoboArm-Vision/                 # ROS 2 workspace 根目录
├── src/                        # ROS 2 功能包
│   ├── arm_interfaces/         # 自定义消息/服务接口
│   ├── arm_utils/              # 跨包通用工具函数
│   ├── arm_control/            # 机械臂通信 + 运动状态机
│   ├── vision_grasp/           # RealSense 视觉 + 抓取检测
│   ├── grasp_pipeline/         # YOLO + SAM + GraspNet 抓取检测流水线
│   ├── llm_voice/              # 大语言模型语音交互
│   └── arm_bringup/            # 启动文件
├── assets/                     # 模型权重（all.pt 已包含，其余需下载）
├── tools/                      # 非 ROS 工具/库
│   ├── eyeInHand/              # 手眼标定
│   ├── graspnet_baseline/      # GraspNet 基线网络、CUDA 算子、API
│   └── test_grasp_pipeline/    # 单帧交互式检测测试
├── config/                     # 共享运行时配置
├── scripts/                    # 常用快捷脚本
├── docs/                       # 详细文档
├── requirements.txt            # Python 依赖
├── README.md
└── CLAUDE.md
```

</details>

---

<a name="features"></a>
## 🚀 核心功能

| 功能模块 | 描述 | 技术栈 |
|----------|------|--------|
| 🔧 **机械臂控制** | 通过 TCP 与下位机通信，实现精确 6D 位姿控制 | Python, ROS 2 |
| 👁️ **视觉识别** | 利用 Intel RealSense 相机进行图像采集与目标检测 | OpenCV, RealSense SDK |
| 📐 **手眼标定** | 棋盘格标定，求解相机到机械臂末端的变换 | OpenCV, transforms3d |
| 🎯 **抓取点检测** | 基于 YOLO + SAM / YOLO-seg + GraspNet 预测最佳抓取位姿 | PyTorch, CNN |
| 🗺️ **运动规划** | 抓取-放置状态机，管理旋转→平移→下降→夹取→放置流程 | 运动学算法 |
| 🗣️ **LLM 语音交互** | 语音/文本输入，智谱 AI 回复，可扩展为控制指令 | ZhipuAI API, 语音识别 |

---

<a name="quickstart"></a>
## 🚀 快速开始

### 1. 安装依赖

```bash
# ROS 2 依赖
sudo apt install ros-humble-realsense2-camera ros-humble-cv-bridge

# Python 依赖
conda create -n grasp python=3.10
conda activate grasp
pip install -r requirements.txt
```

### 2. 下载模型权重

```bash
# 默认下载 GraspNet RealSense 模型 + SAM ViT-B
./scripts/download_assets.sh
```

### 3. 配置 API Key

```bash
cp config/api_keys.yaml.example config/api_keys.yaml
# 编辑 config/api_keys.yaml，填入你的智谱 AI API key
```

### 4. 构建工作空间

```bash
./scripts/build.sh
```

### 5. 启动系统

```bash
# 一键启动核心节点
ros2 launch arm_bringup robot_arm.launch.py

# 或启动语音版
ros2 launch arm_bringup robot_arm_voice.launch.py
```

> 📖 详细步骤、参数调优与故障排查见 [docs/usage.md](docs/usage.md)。

---

<a name="interfaces"></a>
## 🔌 接口与服务

| 名称 | 类型 | 说明 |
|------|------|------|
| `/grasp_result` | `arm_interfaces/msg/GraspResult` | 视觉节点发布的最佳抓取位姿 |
| `/robot_status` | `std_msgs/msg/String` | 运动节点状态反馈，例如 `have backed` |
| `/RobotInfo` | `arm_interfaces/msg/RobotInfo` | 下位机上传的机械臂当前位姿 |
| `/RobotMove` | `trajectory_msgs/msg/JointTrajectory` | 运动指令（下位机接收） |
| `/GripperControl` | `std_msgs/msg/String` | 夹爪控制：`open` / `close` |
| `/llm/ask_text` | `llm_voice/srv/AskText` | 文本问答服务 |
| `/llm/ask_audio` | `std_srvs/srv/Trigger` | 语音问答服务 |

完整消息/服务定义见 `src/arm_interfaces/` 与 `src/llm_voice/srv/`。

---

<a name="docs"></a>
## 📚 详细文档

- **[系统设计文档](docs/architecture.md)**：架构、节点关系、数据流、坐标变换、状态机
- **[使用说明](docs/usage.md)**：安装、配置、启动、调试、常见问题、快捷脚本说明
- **[资产说明](assets/README.md)**：模型权重来源、下载链接、校验建议
- **[CLAUDE.md](CLAUDE.md)**：Claude Code 工作指南与开发约定

### 环境变量

`scripts/` 下的快捷脚本会根据自身位置自动推断仓库根目录，通常无需设置环境变量。如需强制指定：

```bash
export ROBOARM_WS=/path/to/RoboArm-Vision
```

### 模型权重

预训练权重统一放到 `assets/` 目录下：

- `assets/all.pt` —— YOLO 检测 / 分割模型（**已包含在仓库中**）
- `assets/sam_vit_b_01ec64.pth` —— SAM 分割模型（可选）
- `assets/checkpoint.tar` —— GraspNet 抓取模型

> 注意：`assets/` 下的大模型文件已加入 `.gitignore`，不会提交到 Git；`all.pt` 除外。

---

<a name="calibration"></a>
## 📐 手眼标定

使用快捷脚本：

```bash
cd $ROBOARM_WS
./scripts/calib_capture.sh   # 采集约 30 张棋盘格图像
./scripts/calib_run.sh       # 计算相机到机械臂末端的变换
```

或手动运行：

```bash
cd tools/eyeInHand
python3 保存RGB图像.py
python3 eye_in_hand.py
```

---

<a name="history"></a>
## 📅 开发历程

### 🎯 2025 年开发里程碑

| 日期 | 进展 | 状态 |
|------|------|------|
| **12月19日** | 🎨 优化运行效果 | ✅ 完成 |
| **12月18日** | 🔗 完成联调测试 | ✅ 完成 |
| **12月17日** | 🧠 封装大语言模型功能 | ✅ 完成 |
| **12月16日** | 🤖 完成机械臂控制部分测试 | ✅ 完成 |
| **12月15日** | 📡 完成上位机与控制端通信 | ✅ 完成 |
| **12月12日** | 🔧 完成通信自测 & 电控测试 | ✅ 完成 |
| **12月10日** | 📐 完成手眼标定 | ✅ 完成 |
| **12月5日** | ⚙️ 完成手眼标定配置 | ✅ 完成 |
| **12月3日** | 🚀 完成 ORIN NX 移植 | ✅ 完成 |

---

<a name="techstack"></a>
## 🛠️ 技术栈

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/C++-00599C?style=flat&logo=c%2B%2B&logoColor=white" alt="C++">
  <img src="https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white" alt="ROS 2 Humble">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/ZhipuAI-API-orange?style=flat&logo=ai&logoColor=white" alt="ZhipuAI">
</p>

---

<a name="contributing"></a>
## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

- 🔍 发现 bug 或有新需求，请先搜索现有 Issue，避免重复。
- ✅ 提交 PR 前请确保代码可通过 `pytest` 与 `flake8` 基础检查。
- 📝 提交信息使用 Angular 格式，中文简述，例如：
  `feat(scripts): 添加新的调试启动脚本`。

详细开发约定见 [CLAUDE.md](CLAUDE.md)。

---

<a name="license"></a>
## 📄 许可证

本项目采用 [Apache-2.0](LICENSE) 许可证。使用第三方模型权重（如 GraspNet、SAM、YOLO）时，请遵守各权重自身的许可协议。

---

<p align="center">
  ⭐ 如果这个项目对你有帮助，请给个 Star！
</p>
