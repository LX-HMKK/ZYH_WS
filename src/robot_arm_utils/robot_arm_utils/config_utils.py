#coding=utf-8
"""通用配置加载工具。"""
import os
from typing import Any
import yaml


def load_yaml_config(path: str) -> dict:
    """从 YAML 文件加载配置。文件不存在时抛出 FileNotFoundError。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到配置文件：{path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def load_zhipu_api_key(path: str) -> str:
    """从 YAML 配置文件读取智谱 API key。"""
    data = load_yaml_config(path)
    key = data.get("zhipu_api_key", "")
    if not key or key.strip() == "YOUR_ZHIPU_API_KEY_HERE":
        raise ValueError(f"{path} 中 zhipu_api_key 未配置或仍是占位符")
    return key.strip()
