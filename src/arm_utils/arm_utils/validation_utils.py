#coding=utf-8
"""通用验证/解析工具。"""


def is_float(s: str) -> bool:
    """判断字符串是否可以解析为 float。"""
    try:
        float(s)
        return True
    except ValueError:
        return False
