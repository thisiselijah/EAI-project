#!/bin/bash
# 步驟 3：訓練 FLORA DeiT-S 低秩 supernet
# 每個 iteration 隨機抽一個 non-uniform 的 rank 組合（12 blocks × qkv/fc1/fc2，6 種比例 1/6~1）
# 每個 epoch 結束後，用最小的 subnet 在 ImageNet val 上驗證
# 每 5 個 epoch 存一次 checkpoint；中斷後重跑同一個指令，就會從 $OUT_DIR 自動續訓
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$FLORA_DIR"

torchrun --nproc_per_node=$NGPU main.py \
  --cfg configs/lr_deit/supernet/lr_deit_small_supernet.yaml \
  --data-path "$IMAGENET" \
  --batch-size $BATCH \
  --resume "$SUPERNET_INIT" \
  --output "$OUT_DIR" \
  --tag supernet \
  --opts DISTILL.TEACHER_LOGITS_PATH "$LOGITS_DIR" \
         DISTILL.LOGITS_TOPK $LOGITS_TOPK \
         TRAIN.EPOCHS $EPOCHS \
         DATA.NUM_WORKERS $NUM_WORKERS

# 輸出：$OUT_DIR/lr_deit_small_supernet/supernet/ckpt_epoch_*.pth 和 log_rank0.txt
