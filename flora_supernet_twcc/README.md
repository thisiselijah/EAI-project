# 重現論文：在 TWCC 上訓練 FLORA DeiT-S supernet

論文：Chen et al., *Integrating Neural Architecture Search and Rematerialization for Efficient On-Device Learning*, GECCO '25。

## 這一步對應論文的哪裡

論文的框架有兩個輸入：**supernet** 和**目標裝置的資訊**。在 4.3 節和 4.4 節（Raspberry Pi 實驗，Table 4、Table 5），論文直接沿用 FLORA [9] 的 DeiT-S supernet 和 search space（"same search spaces and supernets"）。

所以「訓練 supernet」完全依照 FLORA repo 的流程：

| 步驟 | 腳本 | 內容 |
|---|---|---|
| 0 | `00_setup.sh` | clone FLORA、安裝套件（timm 0.4.12）、下載官方 DeiT-S 權重 |
| 1 | `01_convert_supernet.sh` | 用 SVD 把每個 block 的 qkv / fc1 / fc2 拆成 `U·VT`，轉成 supernet 的初始權重 |
| 2 | `02_save_teacher_logits.sh` | 用 DeiT-S 當 teacher，離線存好每個 epoch 的 top-100 logits（TinyViT 方式） |
| 3 | `03_train_supernet.sh` | 訓練 supernet 300 epochs：每個 iteration 隨機取樣一個 rank 組合（single-path），並用存好的 logits 做蒸餾 |
| – | `run_all.sh` | 依序跑完 0~3，**全部成功後**才用 `twccli rm ccs` 刪除容器 |

**Rockmate 不參與 supernet 訓練。** 它是下一階段才用到的：從 supernet 抽出 subnet 後，對超過記憶體預算 M_budget 的 subnet 做 rematerialization（論文 Algorithm 1 第 9 行）。

Search space（`configs/lr_deit/supernet/lr_deit_small_supernet.yaml`）：

- 12 個 blocks × 3 個線性層（qkv、fc1、fc2）＝ 36 個位置
- 每個位置的 rank 比例有 6 種選擇：1/6, 2/6, …, 1
- 每一層可以選不同的比例（`NON_UNIFORM: True`）
- 論文的 compression baseline「0.167×」，就是所有層都選 1/6 的最小 subnet

## 使用方式（TWCC 8×V100 開發型容器）

1. **建立容器時，選 PyTorch 1.x 的 NGC 映像檔**，例如 `pytorch-22.02-py3`（FLORA 原作者用的就是這版）。
   - FLORA 綁定 `timm==0.4.12`，在 PyTorch 2.x 上可能有相容性問題。
2. 把這個資料夾傳到 HFS，再編輯 `config.sh`：
   - 填入 `<USERNAME>` 和 ImageNet 路徑
   - 想在訓練完自動刪除容器，就填 `CCS_ID`
3. 先單獨跑 `bash 00_setup.sh` 和 `bash 01_convert_supernet.sh`，確認環境沒問題。
4. 開始長時間訓練：
   ```bash
   nohup bash run_all.sh > run_all.log 2>&1 &
   tail -f run_all.log
   ```
5. 觀察訓練進度：
   ```bash
   tail -f $RUN_DIR/output/lr_deit_small_supernet/supernet/log_rank0.txt
   ```
   - 每個 epoch 結束後會印出最小 subnet 的 ImageNet top-1。
   - 程式中斷後，重跑 `run_all.sh` 會從最後一個 checkpoint 自動續訓（每 5 epochs 存一次）。

## 開始前請先確認

- **儲存空間**：teacher logits 每個 epoch 約 0.5 GB，300 epochs 約 150 GB。再加上 checkpoint 和 ImageNet（約 150 GB），請先確認 HFS 配額。
  - 空間不夠時，可以把 `LOGITS_TOPK` 調成 10~20，或把 `EPOCHS` 調低。
  - teacher logits 的 epoch 數必須不少於 supernet 訓練的 epoch 數。腳本會讓兩邊共用同一個 `EPOCHS`。
- **時間和費用（粗估，請依實際速度調整）**：
  - DeiT-S 在 8×V100 上 300 epochs，大約需要 1.5~3 天。
  - 存 teacher logits 是純推論，大約是訓練時間的 1/3。
  - ImageNet 是大量小檔案，HFS 的讀取速度可能成為瓶頸。建議先跑幾百個 iteration，看 log 裡每個 iteration 花多少時間，再推算總時數和 GPU 費用。
- **時程**：如果專題時間緊，可以把 `EPOCHS` 降到 100 左右。這樣 supernet 品質會比較差，但足以跑完整個 NAS 加 rematerialization 流程。
- 總 batch size 是 128 × 8 = 1024。程式會依 `batch × GPU 數 / 512` 自動線性縮放學習率，改 GPU 數量時不需要手動調整。
- 用 `torchrun` 啟動，而不是 README 裡的 `torch.distributed.launch`。FLORA 會從環境變數 `LOCAL_RANK` 讀取 rank，兩種寫法的效果相同。

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
