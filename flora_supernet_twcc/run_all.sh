#!/bin/bash
# 依序跑完步驟 0~3。全部成功才刪除 VCS 虛擬機；任何一步失敗都會保留虛擬機，方便除錯
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
  if [ -n "$VCS_ID" ]; then
    # 結果放在系統碟（掛載點是 /）時，刪除虛擬機會把訓練結果一起刪掉，所以不刪
    if [ "$(df -P "$RUN_DIR" | awk 'NR==2 {print $6}')" = "/" ]; then
      echo "警告：$RUN_DIR 在系統碟上，刪除虛擬機會連訓練結果一起刪掉，所以不刪除。請先把結果搬走，再手動刪除虛擬機" >&2
    else
      echo "刪除 VCS 虛擬機 $VCS_ID（資料磁碟上的結果會保留）"
      $TWCC_CLI_CMD rm vcs -f -s "$VCS_ID"
    fi
  fi
else
  echo "==== [$(date '+%F %T')] 執行失敗，保留虛擬機 ====" >&2
  exit 1
fi
