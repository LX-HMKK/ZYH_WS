# 模型权重与资产说明

本目录存放运行时所需的模型权重与大数据文件。**除 `all.pt` 外，其他权重文件体积过大，不提交到 Git**，请按本说明自行下载或从比赛主办方获取。

## 已包含

| 文件 | 用途 | 说明 |
|------|------|------|
| `all.pt` | YOLO 检测 / 分割模型 | 目标检测与类别识别；若该模型为 YOLO-seg，可直接输出分割掩码，无需再加载 SAM |

## 需自行下载

### 1. GraspNet 抓取模型

本项目使用 [graspnet-baseline](https://github.com/graspnet/graspnet-baseline) 官方预训练权重。
官方提供两个版本：

| 文件 | 适用相机 | Google Drive | 百度网盘 |
|------|----------|--------------|----------|
| `checkpoint-rs.tar` | RealSense D435/D435i | [下载](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing) | [下载](https://pan.baidu.com/s/1Eme60l39tTZrilF0I86R5A) |
| `checkpoint-kn.tar` | Kinect | [下载](https://drive.google.com/file/d/1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX/view?usp=sharing) | [下载](https://pan.baidu.com/s/1QpYzzyID-aG5CgHjPFNB9g) |

**推荐**：使用 RealSense 相机时下载 `checkpoint-rs.tar`，下载后**重命名为 `checkpoint.tar`** 放到本目录。

对应配置项见 `src/grasp_pipeline/config/grasp_config.yaml`：

```yaml
grasp_checkpoint: "{ROBOARM_WS}/assets/checkpoint.tar"
```

### 2. SAM 分割模型（可选）

若 `all.pt` 不是 YOLO-seg 模型，或需要更高质量的分割掩码，可下载 SAM ViT-B：

| 文件 | 来源 | 直接下载 |
|------|------|----------|
| `sam_vit_b_01ec64.pth` | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` |

对应配置项：

```yaml
sam_checkpoint: "{ROBOARM_WS}/assets/sam_vit_b_01ec64.pth"
sam_model_type: "vit_b"
```

如果 `all.pt` 已经是 YOLO-seg 模型，可在 `src/grasp_pipeline/grasp_pipeline/core/object_segmentor.py` 中改用 YOLO 的 seg 输出，并视情况跳过 SAM 加载。

## 快速下载

```bash
cd $ROBOARM_WS
./scripts/download_assets.sh
```

该脚本会尝试自动下载缺失的权重；若无法访问 Google Drive / 百度网盘，会输出手动下载链接。

## 校验（建议）

下载后建议核对文件大小：

| 文件 | 约大小 |
|------|--------|
| `all.pt` | ~40 MB（以实际为准） |
| `checkpoint.tar` | ~100 MB |
| `sam_vit_b_01ec64.pth` | ~375 MB |

> 注：官方权重链接可能随时间变化，如失效请优先到 [graspnet-baseline](https://github.com/graspnet/graspnet-baseline) 和 [segment-anything](https://github.com/facebookresearch/segment-anything) 官方仓库确认最新地址。
