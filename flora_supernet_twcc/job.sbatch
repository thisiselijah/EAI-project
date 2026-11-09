#!/bin/bash
# Slurm 工作內容：在計算節點上依序跑步驟 1~3（不要直接 sbatch 這個檔案，請用 bash submit.sh）
# Slurm 參數（計畫、分割區、GPU 數、時間上限）由 submit.sh 依 config.sh 帶入
# 用法：job.sbatch test  → 只檢查容器裡的 GPU
#       job.sbatch train → 檢查 GPU 後跑步驟 1~3；已完成的步驟會略過，訓練會從最後一個 checkpoint 續訓
set -uo pipefail
HERE="${FLORA_SCRIPTS:?請用 bash submit.sh 送出工作}"
source "$HERE/config.sh"
MODE="${1:-train}"

echo "==== [$(date '+%F %T')] job ${SLURM_JOB_ID:-?} on $(hostname)，模式：$MODE ===="

# 檢查容器裡的 PyTorch 能不能用這台節點的 GPU
in_container python -c "
import torch
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPUs', torch.cuda.device_count())
print('GPU:', torch.cuda.get_device_name(0), '| 支援的架構:', torch.cuda.get_arch_list())
x = torch.randn(4096, 4096, device='cuda')
torch.cuda.synchronize(); print('GPU 運算測試通過', float((x @ x).abs().mean()))
" || { echo "錯誤：容器裡的 PyTorch 無法使用 GPU。請換一個較新的映像檔（config.sh 的 SIF）" >&2; exit 1; }

[ "$MODE" = test ] && exit 0

run() { echo "==== [$(date '+%F %T')] $1 ===="; bash "$HERE/$1"; }

# logits 已存滿 EPOCHS 個 epoch 就跳過步驟 2（方便重送工作）
logits_done() { [ "$(ls -d "$LOGITS_DIR"/*epoch* 2>/dev/null | wc -l)" -ge "$EPOCHS" ]; }

if run 01_convert_supernet.sh \
   && { logits_done && echo "teacher logits 已存在，略過" || run 02_save_teacher_logits.sh; } \
   && run 03_train_supernet.sh; then
  echo "==== [$(date '+%F %T')] 這次工作完成 ===="
else
  echo "==== [$(date '+%F %T')] 執行失敗 ====" >&2
  exit 1
fi
