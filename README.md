# 🤖 机械臂上位机程序

📋 项目简介

本项目是为**机械臂比赛**开发的综合性程序集合，涵盖了**机械臂控制**、**视觉识别**、**抓取点检测**等多个核心模块。作为**2025年机器人学课程**的项目成品，它展示了现代机器人技术的集成应用。

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
│   ├── codroid_msgs/           # 自定义消息
│   ├── grasp_interfaces/       # 自定义服务/话题接口
│   ├── codroid_node/           # 机械臂通信 + 运动状态机
│   │   ├── codroid_node/
│   │   ├── config/
│   │   │   └── motion_config.yaml
│   │   └── ...
│   ├── grasp_publisher/        # RealSense 视觉 + 抓取检测
│   │   ├── grasp_publisher/
│   │   ├── config/
│   │   │   └── grasp_config.yaml
│   │   └── ...
│   ├── llm_voice/              # 大语言模型语音交互
│   └── robot_arm_bringup/      # 启动文件
│       └── launch/
│           └── robot_arm.launch.py
├── assets/                     # 模型权重（大文件，不提交到 Git）
│   ├── all.pt
│   ├── sam_vit_b_01ec64.pth
│   └── ...
├── tools/                      # 非 ROS 工具/库
│   ├── Gloria-M-SDK-1.0.0/     # 夹爪串口 SDK
│   ├── eyeInHand/              # 手眼标定
│   ├── graspnet-baseline-main/ # GraspNet 基线网络
│   ├── librealsense-master/    # RealSense 辅助脚本
│   ├── motor/                  # 电机接口
│   └── label_process/          # 标签处理工具
├── config/                     # 共享运行时配置
│   ├── api_keys.yaml           # 智谱 API key（gitignore）
│   └── api_keys.yaml.example   # 配置模板
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

## 🎯 快速开始

### 环境要求

- Ubuntu 22.04 / ROS 2 Humble
- Python 3.10+
- CUDA 环境（推荐）
- 安装 ROS 2 依赖：

```bash
sudo apt install ros-humble-realsense2-camera ros-humble-cv-bridge
```

### 配置环境变量

在 `~/.bashrc` 中设置仓库根目录：

```bash
export ROBOARM_WS=/home/zyh/ZYH_WS
```

### 构建工作空间

```bash
cd $ROBOARM_WS
conda activate grasp
python -m colcon build --symlink-install
source install/setup.bash
```

### 配置 API Key

```bash
cp config/api_keys.yaml.example config/api_keys.yaml
# 编辑 config/api_keys.yaml 填入智谱 AI 的 API key
```

### 启动系统

一键启动三个核心节点：

```bash
ros2 launch robot_arm_bringup robot_arm.launch.py
```

或分别启动：

```bash
# 终端 1：视觉 / 抓取检测
ros2 run grasp_publisher grasp_node

# 终端 2：上位机通信 + 夹爪控制
ros2 run codroid_node codroid_io

# 终端 3：运动状态机
ros2 run codroid_node codroid_move_test

# 终端 4（可选）：大语言模型交互
ros2 run llm_voice llm_node
```

### 模型权重

预训练权重统一放到 `assets/` 目录下：

- `assets/all.pt` —— YOLOv8 检测模型
- `assets/sam_vit_b_01ec64.pth` —— SAM 分割模型
- `assets/checkpoint.tar` —— GraspNet 抓取模型

> 注意：`assets/` 下的模型文件已加入 `.gitignore`，不会提交到 Git。新克隆仓库后请手动下载或复制。

---

## 📐 手眼标定

进入 `tools/eyeInHand` 目录：

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

---

## 📄 许可证

本项目采用 Apache 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

<div align="center">
  ⭐ 如果这个项目对你有帮助，请给个 Star！
</div>
