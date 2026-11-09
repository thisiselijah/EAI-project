# ===== 共用設定：先改這裡 =====
# 給國網中心晶創主機 Nano5（Slurm 叢集，每個節點 8×H100）使用
# 程式在 Singularity 容器（NVIDIA PyTorch 22.09，內含 PyTorch 1.13）裡執行

export PROJECT_ID=            # ← 填入 iService 計畫 ID，例如 PROJECT_ID=GOV113XXX（用 wallet <計畫ID> 可查餘額）
export DATA_ROOT=/work/$USER  # ← 放資料和輸出的地方。需要約 400 GB 以上的空間

# Slurm 設定（submit.sh 送出工作時使用）
export PARTITION=normal       # normal：H100，最長 48 小時；normal2：H200；dev：測試用，最長 2 小時
export TIME_LIMIT=48:00:00    # 單次工作的時間上限，不能超過分割區的限制
export NGPU=8                 # 每個節點 8 張 GPU
export CPUS=64                # 這個工作要的 CPU 核心數（給 dataloader 用）

# 容器映像檔（系統提供的 NVIDIA PyTorch 22.09）
export SIF=/work/hpc_sys/sifs/pytorch_22.09-py3_horovod.sif

HERE_CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="$(cd "$HERE_CFG/.." && pwd)"
export FLORA_DIR=$REPO_DIR/FLORA                   # EAI-project 的 FLORA 子模組
export IMAGENET=$DATA_ROOT/imagenet                # 底下要有 train/ 和 val/ 兩個資料夾（ImageFolder 格式）
export RUN_DIR=$DATA_ROOT/flora_deit_s             # 權重、logits 和輸出都放這裡
export PYDEPS=$DATA_ROOT/pydeps_flora              # 容器裡額外安裝的套件（timm 0.4.12 等）

export BATCH=128              # 每張卡的 batch；總 batch = 128×8 = 1024，學習率由程式依此線性縮放
export EPOCHS=300             # FLORA 預設值。teacher logits 的 epochs 必須 >= 這個值
export NUM_WORKERS=6          # 每個 GPU process 的 dataloader worker 數；8 × 6 = 48，要小於 CPUS
export LOGITS_TOPK=100        # 每張圖保留 top-k teacher logits（約 0.5 GB/epoch）

# 衍生路徑（不用改）
export DEIT_S_CKPT=$RUN_DIR/deit_small_patch16_224-cd65a155.pth
export SUPERNET_INIT=$RUN_DIR/lr_deit_small_supernet.pth
export LOGITS_DIR=$RUN_DIR/teacher_logits_deit_s
export OUT_DIR=$RUN_DIR/output

# 檢查：資料目錄不存在（例如在自己的電腦上執行）就直接停下來
if [ ! -d "$DATA_ROOT" ] || [ ! -w "$DATA_ROOT" ]; then
  echo "錯誤：找不到可寫入的 $DATA_ROOT。請在 Nano5 上執行，或把 config.sh 的 DATA_ROOT 改成你有寫入權限的目錄" >&2
  exit 1
fi

# 在容器裡執行指令：開啟 GPU（--nv），掛載程式碼和資料目錄，並使用 PYDEPS 裡的套件
export SINGULARITYENV_PYTHONUSERBASE=$PYDEPS
export APPTAINERENV_PYTHONUSERBASE=$PYDEPS
in_container() {
  if ! command -v singularity >/dev/null; then
    [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
    module load singularity || { echo "錯誤：無法載入 singularity 模組（ml singularity）" >&2; return 1; }
  fi
  local binds=(-B "$DATA_ROOT")
  case "$REPO_DIR/" in "$DATA_ROOT"/*) ;; *) binds+=(-B "$REPO_DIR") ;; esac   # 程式碼不在 DATA_ROOT 底下時另外掛載
  singularity exec --nv "${binds[@]}" "$SIF" "$@"
}
