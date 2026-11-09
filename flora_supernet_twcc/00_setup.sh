#!/bin/bash
# 步驟 0：準備 FLORA 程式碼、建立 Python 虛擬環境（PyTorch 1.13 + timm 0.4.12）、下載 DeiT-S 預訓練權重
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

die() { echo "錯誤：$*" >&2; exit 1; }

# 1. GPU 驅動
command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi。請確認這台是 GPU 虛擬機，而且已安裝 NVIDIA 驅動"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

mkdir -p "$RUN_DIR"

# 2. FLORA 程式碼（EAI-project 的子模組）
if [ ! -f "$FLORA_DIR/main.py" ]; then
  echo "FLORA 子模組還沒下載，正在下載"
  git -C "$HERE/.." submodule update --init FLORA
fi

# 3. Python 虛擬環境。FLORA 綁定 timm==0.4.12，所以用 PyTorch 1.13（CUDA 11.7，支援 V100）
if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON" -c 'import sys; sys.exit(0 if (3, 8) <= sys.version_info[:2] <= (3, 10) else 1)' \
    || die "$PYTHON 是 $("$PYTHON" -V 2>&1)，PyTorch 1.13 需要 Python 3.8~3.10。建議建立虛擬機時選 Ubuntu 22.04（內建 Python 3.10），或在 config.sh 把 PYTHON 改成 python3.10"
  "$PYTHON" -m venv "$VENV_DIR" \
    || die "無法建立虛擬環境。Ubuntu 請先執行：sudo apt install -y python3-venv"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r "$FLORA_DIR/requirements.txt"   # timm==0.4.12, yacs, termcolor, wandb

# 4. 下載官方 DeiT-S（ImageNet top-1 79.8%），同時當 supernet 初始化和 teacher
[ -f "$DEIT_S_CKPT" ] || wget -O "$DEIT_S_CKPT" \
  https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

# 5. 檢查環境
python -c "import torch, timm; print('torch', torch.__version__, '| timm', timm.__version__, '| GPUs', torch.cuda.device_count()); assert torch.cuda.is_available(), 'PyTorch 抓不到 GPU'"
for split in train val; do
  [ -d "$IMAGENET/$split" ] || die "找不到 $IMAGENET/$split。請先把 ImageNet 放到資料磁碟（ImageFolder 格式）"
  echo "ImageNet $split 類別數：$(ls "$IMAGENET/$split" | wc -l)（應該是 1000）"
done
df -h "$DATA_ROOT"   # 確認剩餘空間：teacher logits 約需 150 GB
