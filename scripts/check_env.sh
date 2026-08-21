#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

ERR=0
check_pass() { echo "  [OK] $1"; }
check_fail() { echo "  [FAIL] $1" >&2; ERR=1; }
check_warn() { echo "  [WARN] $1" >&2; }

echo "=== 环境检查 ==="
echo "工作空间：$ROBOARM_WS"

# ROS 2
if [ -d /opt/ros/humble ]; then
    check_pass "ROS 2 Humble 已安装"
else
    check_fail "未找到 /opt/ros/humble"
fi

# Conda 环境
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    check_pass "当前 conda 环境：$CONDA_DEFAULT_ENV"
else
    check_warn "未激活 conda 环境（脚本已尝试自动激活 grasp）"
fi

# 工作空间 install
if [ -f "$ROBOARM_WS/install/setup.bash" ]; then
    check_pass "install/setup.bash 已生成"
else
    check_warn "未找到 install/setup.bash，请运行 ./scripts/build.sh"
fi

# 模型权重
echo ""
echo "模型文件检查："
ASSETS_DIR="$ROBOARM_WS/assets"

if [ -f "$ASSETS_DIR/all.pt" ]; then
    check_pass "YOLO 模型：$ASSETS_DIR/all.pt"
else
    check_fail "缺少 YOLO 模型：$ASSETS_DIR/all.pt"
fi

if [ -f "$ASSETS_DIR/checkpoint.tar" ]; then
    check_pass "GraspNet 权重：$ASSETS_DIR/checkpoint.tar"
else
    check_fail "缺少 GraspNet 权重：$ASSETS_DIR/checkpoint.tar"
    echo "       请运行 ./scripts/download_assets.sh 或按 assets/README.md 手动下载"
fi

if [ -f "$ASSETS_DIR/sam_vit_b_01ec64.pth" ]; then
    check_pass "SAM 权重：$ASSETS_DIR/sam_vit_b_01ec64.pth"
else
    check_warn "缺少 SAM 权重（若 all.pt 为 YOLO-seg 可忽略）：$ASSETS_DIR/sam_vit_b_01ec64.pth"
fi

# API key
echo ""
echo "配置文件检查："
if [ -f "$ROBOARM_WS/config/api_keys.yaml" ]; then
    if diff -q "$ROBOARM_WS/config/api_keys.yaml" "$ROBOARM_WS/config/api_keys.yaml.example" >/dev/null 2>&1; then
        check_warn "config/api_keys.yaml 与 example 相同，请填入真实 key"
    else
        check_pass "config/api_keys.yaml 已配置"
    fi
else
    check_warn "未找到 config/api_keys.yaml，将使用 example（LLM 节点可能无法启动）"
fi

# GraspNet 原生算子
echo ""
echo "GraspNet 原生算子检查："
for mod in pointnet2 knn; do
    if python -c "import $mod" 2>/dev/null; then
        check_pass "$mod 可导入"
    else
        check_warn "$mod 未安装，请按 docs/usage.md 2.4 节编译"
    fi
done

echo ""
if [ "$ERR" -eq 0 ]; then
    echo "[OK] 基础环境检查通过"
else
    echo "[FAIL] 存在缺失项，请根据提示补齐"
    exit 1
fi
