#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== 启动带语音控制的抓取系统 ==="
echo "按 Enter 调用 /llm/ask_audio 语音服务，输入 q 退出"

launch_pid=""
trap 'kill_launch_group "$launch_pid"' EXIT

ros2 launch arm_bringup robot_arm_voice.launch.py &
launch_pid=$!

while true; do
    read -r -p "> 按 Enter 开始录音（q 退出）: " key
    if [ "${key:-}" = "q" ]; then
        break
    fi
    ros2 service call /llm/ask_audio std_srvs/srv/Trigger "{}"
done
