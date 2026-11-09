#!/bin/bash
# 步驟 0（在 Nano5 登入節點執行）：準備 FLORA 程式碼、在容器裡安裝 timm 0.4.12 等套件、下載 DeiT-S 預訓練權重
# 這一步只下載和安裝，不做運算；GPU 是否正常請用 bash submit.sh test 檢查
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

die() { echo "錯誤：$*" >&2; exit 1; }

[ -n "$PROJECT_ID" ] || die "請先編輯 config.sh，在 PROJECT_ID= 後面填入 iService 計畫 ID"
[ -f "$SIF" ] || die "找不到容器映像檔 $SIF。請到 /work/hpc_sys/sifs/ 查看目前有哪些映像檔，再修改 config.sh 的 SIF"

mkdir -p "$RUN_DIR" "$PYDEPS"

# 1. FLORA 程式碼（EAI-project 的子模組）
if [ ! -f "$FLORA_DIR/main.py" ]; then
  echo "FLORA 子模組還沒下載，正在下載"
  git -C "$REPO_DIR" submodule update --init FLORA
fi

# 2. 在容器裡安裝 FLORA 需要的套件到 PYDEPS（容器本身已有 PyTorch 1.13 和 torchvision）
in_container pip install --user --no-cache-dir -r "$FLORA_DIR/requirements.txt"   # timm==0.4.12, yacs, termcolor, wandb
in_container python -c "import torch, timm; print('容器內 torch', torch.__version__, '| timm', timm.__version__)"

# 3. 下載官方 DeiT-S（ImageNet top-1 79.8%），同時當 supernet 初始化和 teacher
[ -f "$DEIT_S_CKPT" ] || wget -O "$DEIT_S_CKPT" \
  https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

# 4. 檢查 ImageNet
for split in train val; do
  [ -d "$IMAGENET/$split" ] || die "找不到 $IMAGENET/$split。請先把 ImageNet 上傳到 $IMAGENET（ImageFolder 格式）"
  echo "ImageNet $split 類別數：$(ls "$IMAGENET/$split" | wc -l)（應該是 1000）"
done
df -h "$DATA_ROOT"   # 確認剩餘空間：teacher logits 約需 150 GB

echo "準備完成。接著執行：bash submit.sh test（先測 GPU），沒問題再執行 bash submit.sh"
