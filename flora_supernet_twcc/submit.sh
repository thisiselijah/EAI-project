#!/bin/bash
# 在 Nano5 登入節點送出 Slurm 工作
# 用法：
#   bash submit.sh test      先用 dev 分割區、1 張 GPU、10 分鐘，檢查容器裡的 GPU
#   bash submit.sh           送出訓練工作（步驟 1~3）
#   bash submit.sh train 3   一次送出 3 個接續的工作：前一個結束（例如超過 48 小時上限）後，下一個自動接著續訓
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

[ -n "$PROJECT_ID" ] || { echo "錯誤：請先編輯 config.sh，在 PROJECT_ID= 後面填入 iService 計畫 ID" >&2; exit 1; }
command -v sbatch >/dev/null || { echo "錯誤：找不到 sbatch。請在 Nano5 登入節點上執行" >&2; exit 1; }

MODE="${1:-train}"
COUNT="${2:-1}"
LOG_DIR=$RUN_DIR/slurm_logs
mkdir -p "$LOG_DIR"

common=(--account="$PROJECT_ID" --nodes=1 --ntasks=1
        --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.out"
        --export=ALL,FLORA_SCRIPTS="$HERE")

case "$MODE" in
  test)
    id=$(sbatch --parsable "${common[@]}" --job-name=flora-test --partition=dev \
           --gpus-per-node=1 --cpus-per-task=4 --time=00:10:00 \
           "$HERE/job.sbatch" test)
    echo "已送出 GPU 測試工作 $id"
    ;;
  train)
    prev=
    for i in $(seq 1 "$COUNT"); do
      dep=()
      [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
      prev=$(sbatch --parsable "${common[@]}" ${dep[@]+"${dep[@]}"} --job-name=flora-supernet \
               --partition="$PARTITION" --gpus-per-node="$NGPU" --cpus-per-task="$CPUS" \
               --time="$TIME_LIMIT" "$HERE/job.sbatch" train)
      echo "已送出訓練工作 $prev${dep:+（等前一個工作結束後才開始）}"
    done
    ;;
  *) echo "用法：bash submit.sh [test|train] [工作數]" >&2; exit 1 ;;
esac

echo "查看排隊狀態：squeue -u \$USER"
echo "查看輸出：    tail -f $LOG_DIR/<工作名稱>-<工作ID>.out"
echo "取消工作：    scancel <工作ID>"
