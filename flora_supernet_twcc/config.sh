# ===== 共用設定：先改這裡 =====
# 給台智雲 VCS GPU 虛擬機（例如 vgv.8xsuper，8×V100）使用
# 所有資料都放在 DATA_ROOT。請把它設在另外掛載的資料磁碟（VDS），不要放在系統碟：
#   系統碟容量小，而且刪除虛擬機時會一起被刪掉

export DATA_ROOT=/data        # ← 資料磁碟的掛載點（df -h 可以查看）。需要約 400 GB 以上的空間

HERE_CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FLORA_DIR="$HERE_CFG/../FLORA"              # EAI-project 的 FLORA 子模組
export IMAGENET=$DATA_ROOT/imagenet                # 底下要有 train/ 和 val/ 兩個資料夾（ImageFolder 格式）
export RUN_DIR=$DATA_ROOT/flora_deit_s             # 權重、logits 和輸出都放這裡
export VENV_DIR=$DATA_ROOT/venv_flora              # Python 虛擬環境（PyTorch 1.13 + timm 0.4.12）
export PYTHON=python3                              # 建立虛擬環境用的 Python，必須是 3.8~3.10

export NGPU=8                 # 8×V100
export BATCH=128              # 每張卡的 batch；總 batch = 128×8 = 1024，學習率由程式依此線性縮放
export EPOCHS=300             # FLORA 預設值。teacher logits 的 epochs 必須 >= 這個值
export NUM_WORKERS=4          # 每個 GPU process 的 dataloader worker 數；CPU 不夠時再調低
export LOGITS_TOPK=100        # 每張圖保留 top-k teacher logits（約 0.5 GB/epoch）

# 全部跑完後自動刪除虛擬機用（twccli ls vcs 查到的 ID）；留空就不會刪
# 只有 RUN_DIR 在系統碟以外的磁碟時才會真的刪除，避免訓練結果跟著虛擬機一起消失
export VCS_ID=
export TWCC_CLI_CMD=twccli

# 衍生路徑（不用改）
export DEIT_S_CKPT=$RUN_DIR/deit_small_patch16_224-cd65a155.pth
export SUPERNET_INIT=$RUN_DIR/lr_deit_small_supernet.pth
export LOGITS_DIR=$RUN_DIR/teacher_logits_deit_s
export OUT_DIR=$RUN_DIR/output

# 檢查：資料磁碟不存在（例如在自己的電腦上執行）就直接停下來
if [ ! -d "$DATA_ROOT" ] || [ ! -w "$DATA_ROOT" ]; then
  echo "錯誤：找不到可寫入的 $DATA_ROOT。請在 VCS GPU 虛擬機上執行，並把 config.sh 的 DATA_ROOT 改成資料磁碟的掛載點" >&2
  exit 1
fi

# 00_setup.sh 建好虛擬環境後，其他步驟會自動使用它
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
fi
