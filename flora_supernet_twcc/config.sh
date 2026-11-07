# ===== 共用設定：先改這裡 =====
# 路徑都放在 HFS（/home/<帳號> 或 /work/<帳號>），容器刪掉後資料仍會保留

export FLORA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/FLORA"   # 預設指向 EAI-project 的 FLORA 子模組
export IMAGENET=/work/<USERNAME>/imagenet          # 底下要有 train/ 和 val/ 兩個資料夾（ImageFolder 格式）
export RUN_DIR=/work/<USERNAME>/flora_deit_s       # 權重、logits 和輸出都放這裡

export NGPU=8                 # 8×V100
export BATCH=128              # 每張卡的 batch；總 batch = 128×8 = 1024，學習率由程式依此線性縮放
export EPOCHS=300             # FLORA 預設值。teacher logits 的 epochs 必須 >= 這個值
export NUM_WORKERS=4          # 每個 GPU process 的 dataloader worker 數；8 卡容器 CPU 不夠時再調低
export LOGITS_TOPK=100        # 每張圖保留 top-k teacher logits（約 0.5 GB/epoch）

# 自動刪除容器用（twccli ls ccs 查到的 ID）；留空就不會刪
export CCS_ID=
export TWCC_CLI_CMD=/home/<USERNAME>/.local/bin/twccli

# 衍生路徑（不用改）
export DEIT_S_CKPT=$RUN_DIR/deit_small_patch16_224-cd65a155.pth
export SUPERNET_INIT=$RUN_DIR/lr_deit_small_supernet.pth
export LOGITS_DIR=$RUN_DIR/teacher_logits_deit_s
export OUT_DIR=$RUN_DIR/output
