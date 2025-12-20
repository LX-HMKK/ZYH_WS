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
├── src/ # 源代码主目录 
│ ├── Gloria-M-SDK-1.0.0/ # 机械臂控制SDK 
│ │ ├── examples/ # SDK使用示例 
│ │ ├── gloria_msdk/ # 核心SDK代码 
│ │ │ ├── core/ # 核心功能模块 
│ │ │ ├── models/ # 数据模型 
│ │ ├── motor/ # 电机控制相关 
│ │ ├── tests/ # 单元测试 
│ │ └── ... 
│ ├── eyeInHand/ # 手眼标定相关代码 
│ ├── grasp_ws/ # ROS工作空间，包含抓取相关节点 
│ │ ├── config/ # 配置文件 
│ │ ├── src/ # ROS源代码 
│ │ │ ├── codroid_node/ # 主控制节点 
│ │ │ ├── grasp_publisher/ # 抓取信息发布节点 
│ │ │ └── llm_voice/ # 大语言模型语音交互节点 
│ │ └── ... │ ├── graspnet-baseline-main/ # 基于深度学习的抓取检测网络 
│ │ ├── dataset/ # 数据集处理工具 
│ │ ├── models/ # 网络模型定义 
│ │ ├── kw/ # 关键算法实现 
│ │ ├── zyh_code/ # 自定义实现代码 
│ │ └── ... 
│ ├── librealsense-master/ # Intel RealSense相机驱动及处理代码 
│ └── motor/ # 电机控制接口 
├── eyeInHand/ # 手眼标定独立目录 
├── label_process/ # 标签处理工具 
└── ...
```

---

## 🚀 核心功能

| 功能模块            | 描述                              | 技术栈                   |
| --------------- | ------------------------------- | --------------------- |
| 🔧 **机械臂控制**    | 通过Gloria-M-SDK实现精确的机械臂运动控制      | Python, ROS              |
| 👁️ **视觉识别**    | 利用Intel RealSense相机进行高精度图像采集和处理 | OpenCV, RealSense SDK |
| 📐 **手眼标定**     | 实现相机坐标系与机械臂坐标系的精确转换             | 计算机视觉算法               |
| 🎯 **抓取点检测**    | 基于深度学习算法智能检测物体的最佳抓取位置           | PyTorch, CNN          |
| 🗺️ **运动规划**    | 智能路径规划，确保抓取任务的顺利执行              | 运动学算法                 |
| 🗣️ **LLM语音交互** | 通过大语言模型实现语音交互，支持机械臂控制指令和物品信息查询  | ZhipuAI API, 语音识别     |

---

## 🎯 快速开始

### 🏃‍♂️ 主运行节点

进入 **`grasp_ws`** 目录，查看 [详细使用说明](grasp_ws/readme.md)

### 📐 手眼标定

进入 **`eyeInHand`** 目录，查看 [标定指南](eyeInHand/readme.md)

### 🧠 物品识别模型

预训练YOLOv8模型位置：`src/graspnet-baseline-main/all.pt`

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

欢迎提交Issue和Pull Request来改进项目！

---

## 📄 许可证

本项目采用Apache许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

<div align="center">
  ⭐ 如果这个项目对你有帮助，请给个Star！
</div>
