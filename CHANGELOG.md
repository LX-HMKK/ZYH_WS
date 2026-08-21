# 更新日志

本文件记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并遵循 [语义化版本](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## [未发布] - 2026-08-21

### 新增
- 添加标定、单次抓取、循环抓取、语音抓取等快捷启动脚本
- 添加单帧 YOLO + SAM + GraspNet 交互式检测测试
- 添加语音模式启动文件 `robot_arm_voice.launch.py`
- 添加模型权重下载脚本 `scripts/download_assets.sh`
- 添加构建、清理、环境检查脚本
- 添加根目录 `requirements.txt` 与 `.gitattributes`

### 修复
- 修复损坏的 `src/llm_voice/srv/AskText.srv`
- 补全多个功能包的 `package.xml` 依赖声明

## [0.0.1] - 2025-12-19

### 历史里程碑
- 12月19日：优化运行效果
- 12月18日：完成联调测试
- 12月17日：封装大语言模型功能
- 12月16日：完成机械臂控制部分测试
- 12月15日：完成上位机与控制端通信
- 12月12日：完成通信自测 & 电控测试
- 12月10日：完成手眼标定
- 12月5日：完成手眼标定配置
- 12月3日：完成 ORIN NX 移植
