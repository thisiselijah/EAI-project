#!/bin/bash
# 步驟 2：離線知識蒸餾（TinyViT 方式），先用 DeiT-S teacher 存好每個 epoch 的 top-k logits
# 每個 epoch 約 0.5 GB，300 epochs 約 150 GB → 先確認 DATA_ROOT 的空間
# 由 job.sbatch 在計算節點上呼叫
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$FLORA_DIR"

in_container python -m torch.distributed.run --nproc_per_node=$NGPU save_logits.py \
  --cfg configs/teacher/deit_s.yaml \
  --data-path "$IMAGENET" \
  --batch-size $BATCH \
  --resume "$DEIT_S_CKPT" \
  --skip-eval \
  --output "$RUN_DIR/teacher_log" \
  --opts DISTILL.TEACHER_LOGITS_PATH "$LOGITS_DIR" \
         DISTILL.LOGITS_TOPK $LOGITS_TOPK \
         TRAIN.EPOCHS $EPOCHS \
         DATA.NUM_WORKERS $NUM_WORKERS

# （選用）抽查存好的 logits 是否和 teacher 重新推論的結果一致：
# in_container python -m torch.distributed.run --nproc_per_node=$NGPU save_logits.py --cfg configs/teacher/deit_s.yaml \
#   --data-path "$IMAGENET" --batch-size $BATCH --resume "$DEIT_S_CKPT" --check-saved-logits \
#   --opts DISTILL.TEACHER_LOGITS_PATH "$LOGITS_DIR" TRAIN.EPOCHS 1
