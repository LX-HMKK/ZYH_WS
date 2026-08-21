#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== 启动完整一次抓取 ==="
echo "视觉 / IO / 运动节点已启动，完成一次抓取-放置后自动停止"

launch_pid=""
trap 'kill_launch_group "$launch_pid"' EXIT

ros2 launch arm_bringup robot_arm.launch.py &
launch_pid=$!

python3 "$SCRIPT_DIR/_helpers/wait_once.py" 120
