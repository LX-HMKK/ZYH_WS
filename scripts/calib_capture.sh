#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== 启动标定图像采集 ==="
echo "按 's' 保存当前帧，按 'q' 或 ESC 退出"

cd "$ROBOARM_WS/tools/eyeInHand"
python3 保存RGB图像.py
