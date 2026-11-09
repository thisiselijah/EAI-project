# 重現論文：在國網晶創主機 Nano5 上訓練 FLORA DeiT-S supernet

論文：Chen et al., *Integrating Neural Architecture Search and Rematerialization for Efficient On-Device Learning*, GECCO '25。

## 這一步對應論文的哪裡

論文的框架有兩個輸入：**supernet** 和**目標裝置的資訊**。在 4.3 節和 4.4 節（Raspberry Pi 實驗，Table 4、Table 5），論文直接沿用 FLORA [9] 的 DeiT-S supernet 和 search space（"same search spaces and supernets"）。

所以「訓練 supernet」完全依照 FLORA repo 的流程：

| 步驟 | 腳本 | 內容 |
|---|---|---|
| 0 | `00_setup.sh` | 在登入節點執行：準備 FLORA 子模組、在容器裡安裝 timm 0.4.12 等套件、下載官方 DeiT-S 權重 |
| 1 | `01_convert_supernet.sh` | 用 SVD 把每個 block 的 qkv / fc1 / fc2 拆成 `U·VT`，轉成 supernet 的初始權重 |
| 2 | `02_save_teacher_logits.sh` | 用 DeiT-S 當 teacher，離線存好每個 epoch 的 top-100 logits（TinyViT 方式） |
| 3 | `03_train_supernet.sh` | 訓練 supernet 300 epochs：每個 iteration 隨機取樣一個 rank 組合（single-path），並用存好的 logits 做蒸餾 |
| – | `submit.sh` | 在登入節點把 `job.sbatch` 送給 Slurm；`test` 模式只檢查 GPU |
| – | `job.sbatch` | Slurm 工作內容：在計算節點上依序跑 1~3，已完成的步驟會略過 |

**Rockmate 不參與 supernet 訓練。** 它是下一階段才用到的：從 supernet 抽出 subnet 後，對超過記憶體預算 M_budget 的 subnet 做 rematerialization（論文 Algorithm 1 第 9 行）。

Search space（`configs/lr_deit/supernet/lr_deit_small_supernet.yaml`）：

- 12 個 blocks × 3 個線性層（qkv、fc1、fc2）＝ 36 個位置
- 每個位置的 rank 比例有 6 種選擇：1/6, 2/6, …, 1
- 每一層可以選不同的比例（`NON_UNIFORM: True`）
- 論文的 compression baseline「0.167×」，就是所有層都選 1/6 的最小 subnet

## 使用方式（國網晶創主機 Nano5）

> 台智雲已在 2026-08-30 停止提供開發型容器（CCS）和 HFS，這裡改用國網中心的晶創主機 Nano5。
> Nano5 是 Slurm 叢集，每個節點 8×H100（或 H200）。程式在系統提供的 Singularity 映像檔
> `pytorch_22.09-py3`（PyTorch 1.13，符合 FLORA 綁定的 `timm==0.4.12`）裡執行。
> 說明文件：https://man.twcc.ai/@AI-Pilot/manual

1. **準備帳號**：iService 會員帳號、加入有 Nano5 額度的計畫，並設定主機帳號、主機密碼和 OTP（IDExpert App）。
2. **登入**（連續輸錯 3 次會被鎖 15 分鐘）：
   ```bash
   ssh <主機帳號>@nano5.nchc.org.tw
   ```
3. **上傳 ImageNet**：放到 `/work/<主機帳號>/imagenet`（ImageFolder 格式，`train/`、`val/` 各 1000 個類別）。傳檔走 port 2222：
   ```bash
   scp -P 2222 -r <本機的 imagenet 資料夾> <主機帳號>@nano5.nchc.org.tw:/work/<主機帳號>/
   ```
4. **下載程式碼並設定**：
   ```bash
   cd /work/$USER
   git clone --recurse-submodules -b supernet-scripts https://github.com/thisiselijah/EAI-project.git
   cd EAI-project/flora_supernet_twcc
   ```
   編輯 `config.sh`：
   - `PROJECT_ID` 填入計畫 ID（例如 `GOV113XXX`）。
   - `DATA_ROOT` 預設是 `/work/$USER`，要放在別處再改。
5. **準備環境**（在登入節點執行，只下載和安裝）：
   ```bash
   bash 00_setup.sh
   ```
6. **測試 GPU**：用 `dev` 分割區、1 張 GPU 跑 10 分鐘以內，確認容器裡的 PyTorch 能用 H100：
   ```bash
   bash submit.sh test
   ```
   輸出在 `$RUN_DIR/slurm_logs/`，最後要看到「GPU 運算測試通過」。
7. **送出訓練工作**：
   ```bash
   bash submit.sh
   ```
   `normal` 分割區單次最長 48 小時。怕跑不完，可以一次送出多個接續的工作，前一個結束後下一個會自動續訓：
   ```bash
   bash submit.sh train 3
   ```
8. **觀察進度**：
   ```bash
   squeue -u $USER
   tail -f $RUN_DIR/output/lr_deit_small_supernet/supernet/log_rank0.txt
   ```
   - 每個 epoch 結束後會印出最小 subnet 的 ImageNet top-1。
   - 工作中斷後，重新 `bash submit.sh` 會從最後一個 checkpoint 自動續訓（每 5 epochs 存一次）。
   - 不要的工作用 `scancel <工作ID>` 取消。工作在 RUNNING 狀態就會扣計畫錢包的 SU，可以用 `wallet <計畫ID>` 查餘額。

## 開始前請先確認

- **儲存空間**：teacher logits 每個 epoch 約 0.5 GB，300 epochs 約 150 GB。再加上 checkpoint 和 ImageNet（約 150 GB），`DATA_ROOT` 至少要有 400 GB 可用空間。
  - 空間不夠時，可以把 `LOGITS_TOPK` 調成 10~20，或把 `EPOCHS` 調低。
  - teacher logits 的 epoch 數必須不少於 supernet 訓練的 epoch 數。腳本會讓兩邊共用同一個 `EPOCHS`。
- **時間和費用（粗估，請依實際速度調整）**：
  - DeiT-S 在 8×V100 上 300 epochs 大約需要 1.5~3 天；H100 快很多，實際時間請看 log 推算。
  - 存 teacher logits 是純推論，大約是訓練時間的 1/3。
  - ImageNet 是大量小檔案，檔案系統的讀取速度可能成為瓶頸。建議先跑幾百個 iteration，看 log 裡每個 iteration 花多少時間，再推算總時數和 GPU 費用。
- **時程**：如果專題時間緊，可以把 `EPOCHS` 降到 100 左右。這樣 supernet 品質會比較差，但足以跑完整個 NAS 加 rematerialization 流程。
- 總 batch size 是 128 × 8 = 1024。程式會依 `batch × GPU 數 / 512` 自動線性縮放學習率，改 GPU 數量時不需要手動調整。
- 用 `python -m torch.distributed.run`（也就是 `torchrun`）啟動，而不是 FLORA README 裡的 `torch.distributed.launch`。FLORA 會從環境變數 `LOCAL_RANK` 讀取 rank，兩種寫法的效果相同。

## 訓練完之後（論文的下一階段）

1. **抽出 subnet**：
   ```bash
   python subnet.py --cfg <subnet yaml> --pretrained <ckpt>
   ```
   用這個指令把 compression baseline（全部選 1/6）和候選 subnet 從 supernet 切出來。
2. **整合 Rockmate**（`pip install rockmate`）：
   - 對每個 subnet 量測訓練時的峰值記憶體 M 和訓練時間 T_train。
   - M 超過 M_budget 的 subnet，用 Rockmate 產生 remat 版本，再重新量測時間。
   - 超過 T_limit 的就丟掉。T_limit 是 full-rank DeiT-S 只用 remat 的訓練時間。
3. **演化搜尋**：
   - 把 `evolution_search.py` 的 fitness 改成論文 Eq. 3 的 `Acc_val / T_train`，限制條件改成 M_budget。
   - 原本的 fitness 和限制是 GFLOPs。
