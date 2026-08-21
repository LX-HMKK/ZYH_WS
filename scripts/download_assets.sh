#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

ASSETS_DIR="$ROBOARM_WS/assets"
mkdir -p "$ASSETS_DIR"

echo "=== 模型权重下载脚本 ==="
echo "目标目录：$ASSETS_DIR"

# SAM 官方直链
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_FILE="$ASSETS_DIR/sam_vit_b_01ec64.pth"

# GraspNet 官方 Google Drive 文件 ID
GRASPNET_RS_ID="1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk"
GRASPNET_KN_ID="1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX"

# 默认下载 RealSense 版本
GRASPNET_VARIANT="${1:-rs}"
case "$GRASPNET_VARIANT" in
    rs|RS)
        GRASPNET_ID="$GRASPNET_RS_ID"
        GRASPNET_SRC="checkpoint-rs.tar"
        ;;
    kn|KN)
        GRASPNET_ID="$GRASPNET_KN_ID"
        GRASPNET_SRC="checkpoint-kn.tar"
        ;;
    *)
        echo "用法：$0 [rs|kn]"
        echo "  rs - RealSense 模型（默认）"
        echo "  kn - Kinect 模型"
        exit 1
        ;;
esac

GRASPNET_OUT="$ASSETS_DIR/checkpoint.tar"

download_file() {
    local url="$1"
    local out="$2"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$out" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - -o "$out" "$url"
    else
        echo "[ERROR] 未找到 wget 或 curl，无法下载 $url" >&2
        return 1
    fi
}

# 下载 SAM（可选，若 all.pt 为 YOLO-seg 可跳过）
if [ -f "$SAM_FILE" ]; then
    echo "[SKIP] SAM 权重已存在：$SAM_FILE"
else
    echo "[DOWNLOAD] 正在下载 SAM ViT-B 权重..."
    download_file "$SAM_URL" "$SAM_FILE"
    echo "[OK] SAM 权重已保存：$SAM_FILE"
fi

# 下载 GraspNet
if [ -f "$GRASPNET_OUT" ]; then
    echo "[SKIP] GraspNet 权重已存在：$GRASPNET_OUT"
else
    if command -v gdown >/dev/null 2>&1; then
        echo "[DOWNLOAD] 正在通过 gdown 下载 GraspNet $GRASPNET_SRC ..."
        gdown "$GRASPNET_ID" -O "$GRASPNET_OUT"
        echo "[OK] GraspNet 权重已保存：$GRASPNET_OUT"
    else
        echo "[WARN] 未安装 gdown，无法自动从 Google Drive 下载"
        echo "请手动下载 $GRASPNET_SRC 并重命名为 checkpoint.tar 后放到："
        echo "  $GRASPNET_OUT"
        echo "Google Drive: https://drive.google.com/file/d/$GRASPNET_ID/view?usp=sharing"
        echo "百度网盘链接见 assets/README.md"
    fi
fi

echo ""
echo "=== 下载完成 ==="
ls -lh "$ASSETS_DIR"
