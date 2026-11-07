#!/bin/bash
# 步驟 0：下載 FLORA、安裝套件、下載 DeiT-S 預訓練權重
set -euo pipefail
source "$(dirname "$0")/config.sh"

mkdir -p "$RUN_DIR"
[ -d "$FLORA_DIR" ] || git clone https://github.com/shadowpa0327/FLORA "$FLORA_DIR"

cd "$FLORA_DIR"
pip install --user -r requirements.txt   # timm==0.4.12, yacs, termcolor, wandb

# 下載官方 DeiT-S（ImageNet top-1 79.8%），同時當 supernet 初始化和 teacher
[ -f "$DEIT_S_CKPT" ] || wget -O "$DEIT_S_CKPT" \
  https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

# 檢查環境
python -c "import torch, timm; print('torch', torch.__version__, '| timm', timm.__version__, '| GPUs', torch.cuda.device_count())"
ls "$IMAGENET/train" | wc -l   # 應該是 1000
ls "$IMAGENET/val"   | wc -l   # 應該是 1000
