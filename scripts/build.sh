#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== 构建工作空间 ==="
echo "工作空间：$ROBOARM_WS"

python -m colcon build --symlink-install "$@"

# shellcheck source=/dev/null
source "$ROBOARM_WS/install/setup.bash"

echo ""
echo "[OK] 构建完成，install/setup.bash 已重新 source"
