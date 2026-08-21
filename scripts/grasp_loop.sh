#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== 启动多次抓取（直至无目标）==="
echo "完成至少一次抓取后，若 20s 未检测到新目标则自动停止"
echo "也可随时按 Ctrl+C 结束"

launch_pid=""
trap 'kill_launch_group "$launch_pid"' EXIT

ros2 launch arm_bringup robot_arm.launch.py &
launch_pid=$!

python3 "$SCRIPT_DIR/_helpers/wait_loop.py" 20 600
