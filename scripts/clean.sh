#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== 清理工作空间构建产物 ==="
echo "工作空间：$ROBOARM_WS"

cd "$ROBOARM_WS"

# 删除 colcon 构建产物
rm -rf build install log

# 删除 Python 缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "[OK] 已清理 build/ install/ log/ 与 Python 缓存"
