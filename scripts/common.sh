#!/usr/bin/env bash
# 机械臂快捷脚本公共环境初始化
# 被 scripts/ 下的其他脚本 source 使用

set -euo pipefail

# 脚本目录（供调用者使用）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 工作空间根目录：
# 1. 优先使用环境变量 ROBOARM_WS；
# 2. 未设置时，根据本脚本位置自动推断仓库根目录（scripts/ 的上一级）。
if [ -n "${ROBOARM_WS:-}" ]; then
    export ROBOARM_WS
elif [ -d "$SCRIPT_DIR/../src" ] && [ -d "$SCRIPT_DIR/../tools" ]; then
    export ROBOARM_WS="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    export ROBOARM_WS="/home/zyh/ZYH_WS"
fi

if [ ! -d "$ROBOARM_WS" ]; then
    echo "[ERROR] 工作空间 $ROBOARM_WS 不存在，请设置 ROBOARM_WS 环境变量" >&2
    exit 1
fi

cd "$ROBOARM_WS"
if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck source=/dev/null
    source /opt/ros/humble/setup.bash
else
    echo "[WARN] 未找到 /opt/ros/humble/setup.bash" >&2
fi

# 工作空间 install
if [ -f "$ROBOARM_WS/install/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "$ROBOARM_WS/install/setup.bash"
else
    echo "[WARN] 未找到 $ROBOARM_WS/install/setup.bash，请先 colcon build" >&2
fi

# 激活 conda 环境（若已激活则跳过）
if [ -z "${CONDA_DEFAULT_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate grasp 2>/dev/null || echo "[WARN] 无法自动激活 conda 环境 grasp" >&2
fi

# 终止 ros2 launch 进程组（通过 PGID 级联杀掉所有子节点）
kill_launch_group() {
    local pid="$1"
    [ -z "$pid" ] && return 0
    local pgid
    pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ') || true
    [ -z "$pgid" ] && return 0
    kill -TERM -- -"$pgid" 2>/dev/null || true
    sleep 1
    kill -KILL -- -"$pgid" 2>/dev/null || true
}
