#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== 测试识别与 GraspNet 结果生成（带 imshow）==="
echo "相机预览中按 's' 保存并检测，按 'q' 或 ESC 退出"

python3 "$ROBOARM_WS/tools/test_grasp_pipeline/test_detection_imshow.py"
