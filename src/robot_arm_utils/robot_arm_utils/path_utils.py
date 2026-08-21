#coding=utf-8
"""路径工具函数。"""
import os


def get_workspace_root() -> str:
    """返回仓库根目录，优先读取 ROBOARM_WS 环境变量。"""
    return os.environ.get("ROBOARM_WS", "/home/zyh/ZYH_WS")
