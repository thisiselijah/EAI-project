#!/bin/bash
# 步驟 1：用 SVD 把 DeiT-S 的 qkv / fc1 / fc2 拆成低秩形式（U、VT），轉成 supernet 的初始權重
# 只用 CPU 跑，幾分鐘內完成
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$FLORA_DIR"

if [ ! -f "$SUPERNET_INIT" ]; then
  python supernet.py \
    --cfg configs/lr_deit/supernet/lr_deit_small_supernet.yaml \
    --pretrained "$DEIT_S_CKPT"
  # supernet.py 會把檔案輸出到目前目錄，檔名是 <MODEL.NAME>.pth
  mv lr_deit_small_supernet.pth "$SUPERNET_INIT"
fi
ls -lh "$SUPERNET_INIT"
