#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== 开始手眼标定 ==="

cd "$ROBOARM_WS/tools/eyeInHand"
python3 eye_in_hand.py
