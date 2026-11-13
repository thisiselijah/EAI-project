#!/bin/bash
# 從 Kaggle 下載 ImageNet-1k（ILSVRC2012），整理成 FLORA 需要的 ImageFolder 格式：
#   $DEST/train/<wnid>/*.JPEG   (1000 類, 1,281,167 張)
#   $DEST/val/<wnid>/*.JPEG     (1000 類, 50,000 張)
#
# 事前準備（只做一次）：
#   1. 到 https://www.kaggle.com/c/imagenet-object-localization-challenge/rules 按 "I Understand and Accept"
#   2. Kaggle → Settings → API → Create New Token，下載 kaggle.json 後放到主機：
#        mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   3. pip install --user kaggle
#
# 空間：zip 約 160 GB + 解壓後約 160 GB，處理期間需要約 320 GB；確認無誤後再手動刪除 $TMP。
# 用法：bash prepare_imagenet.sh            （或包成 sbatch 送到 CPU 節點跑，見 README）
set -euo pipefail

DEST=${DEST:-/work/$USER/imagenet}
TMP=${TMP_DIR:-/work/$USER/imagenet_raw}
mkdir -p "$DEST" "$TMP"
cd "$TMP"

echo "[$(date '+%F %T')] 1/4 下載（約 160 GB，可能要數小時）"
[ -f imagenet-object-localization-challenge.zip ] || \
  kaggle competitions download -c imagenet-object-localization-challenge -p "$TMP"

echo "[$(date '+%F %T')] 2/4 解壓縮"
[ -d ILSVRC/Data/CLS-LOC/train ] || unzip -q imagenet-object-localization-challenge.zip

echo "[$(date '+%F %T')] 3/4 整理 train/"
[ -d "$DEST/train" ] || mv ILSVRC/Data/CLS-LOC/train "$DEST/train"

echo "[$(date '+%F %T')] 4/4 依 LOC_val_solution.csv 把 val/ 分到 1000 個類別資料夾"
if [ ! -d "$DEST/val" ]; then
  mkdir -p "$DEST/val"
  python3 - "$TMP" "$DEST/val" <<'EOF'
import csv, os, sys, shutil
tmp, out = sys.argv[1], sys.argv[2]
src = os.path.join(tmp, 'ILSVRC/Data/CLS-LOC/val')
with open(os.path.join(tmp, 'LOC_val_solution.csv')) as f:
    for row in csv.DictReader(f):
        wnid = row['PredictionString'].split()[0]
        os.makedirs(os.path.join(out, wnid), exist_ok=True)
        shutil.move(os.path.join(src, row['ImageId'] + '.JPEG'),
                    os.path.join(out, wnid, row['ImageId'] + '.JPEG'))
EOF
fi

echo "== 檢查 =="
echo "train 類別數: $(ls "$DEST/train" | wc -l)  (應為 1000)"
echo "val   類別數: $(ls "$DEST/val"   | wc -l)  (應為 1000)"
echo "val   張數:   $(find "$DEST/val" -name '*.JPEG' | wc -l)  (應為 50000)"

# 確認無誤後再刪除原始檔：rm -rf "$TMP"
echo "完成。確認數量正確後可執行：rm -rf $TMP"
