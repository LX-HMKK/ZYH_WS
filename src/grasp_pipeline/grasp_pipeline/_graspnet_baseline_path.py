#coding=utf-8
"""GraspNet 基线库路径 shim。

原 GraspNet 代码通过顶层包名 `graspnet`、`graspnetAPI`、`data_utils`、
`collision_detector` 导入，必须将 tools/graspnet_baseline/models 与
utils 加入 sys.path 才能兼容。本模块把这些副作用集中在一处，
由真正需要 GraspNet 导入的模块显式导入。
"""
import os
import sys

from arm_utils import get_workspace_root


_WORKSPACE_ROOT = get_workspace_root()
sys.path.append(os.path.join(_WORKSPACE_ROOT, "tools", "graspnet_baseline", "models"))
sys.path.append(os.path.join(_WORKSPACE_ROOT, "tools", "graspnet_baseline", "utils"))
