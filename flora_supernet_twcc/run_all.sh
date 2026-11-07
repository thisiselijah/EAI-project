#!/bin/bash
# 依序跑完步驟 0~3。全部成功才刪除 TWCC 容器；任何一步失敗都會保留容器，方便除錯
# 用法（SSH 斷線也會繼續跑）：
#   nohup bash run_all.sh > run_all.log 2>&1 &
#   tail -f run_all.log
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

run() { echo "==== [$(date '+%F %T')] $1 ===="; bash "$HERE/$1"; }

# logits 已存滿 EPOCHS 個 epoch 就跳過步驟 2（方便重跑）
logits_done() { [ "$(ls -d "$LOGITS_DIR"/*epoch* 2>/dev/null | wc -l)" -ge "$EPOCHS" ]; }

if run 00_setup.sh \
   && run 01_convert_supernet.sh \
   && { logits_done && echo "teacher logits 已存在，略過" || run 02_save_teacher_logits.sh; } \
   && run 03_train_supernet.sh; then
  echo "==== [$(date '+%F %T')] 全部完成 ===="
  if [ -n "$CCS_ID" ]; then
    echo "刪除開發型容器 $CCS_ID"
    $TWCC_CLI_CMD rm ccs -f -s "$CCS_ID"
  fi
else
  echo "==== [$(date '+%F %T')] 執行失敗，保留容器 ====" >&2
  exit 1
fi
