# EAI Final Project：Raspberry Pi 上的持續學習系統

在 Raspberry Pi 5（1 GB 訓練記憶體預算）上，結合 **NAS + rematerialization**（Chen et al., GECCO 2025）與 **Latent Replay**（Pellegrini et al., IROS 2020），在 CORe50 串流資料上持續學習新物品。

## Quick Start 

```bash
git clone --recursive https://github.com/thisiselijah/EAI-project.git
cd EAI-project
```

### 安裝環境（uv）

用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本、虛擬環境和套件（設定在 `pyproject.toml`，Python 版本在 `.python-version`）：

```bash
uv sync                  # Create .venv and install dependencies
uv sync --extra remat    # Install rockmate submodule（rematerialization，需要 C 編譯器）
uv run python autoformer_cl/subnet_split.py -l 9
```

第一次 `uv sync` 會產生 `uv.lock`，請一起 commit。

### 下載預訓練權重（不放在 git 裡）

把權重放在 repo **外面**的 `../weights/`：

```bash
mkdir -p ../weights && cd ../weights
# AutoFormer 官方 supernet-S（139 MB）
wget https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-small.pth
# 官方 DeiT-S（88 MB，選用）
wget https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
cd ../EAI-project
```

### 測試：切出子網路並在第 l 層分段

```bash
uv run python autoformer_cl/subnet_split.py -l 9   # l = latent layer，可試 6 / 9 / 11
```

## 專案結構

| 路徑 | 內容 |
|---|---|
| `autoformer_cl/` | 我們的程式：從 AutoFormer supernet 切出子網路、Frozen / Plastic 分段（latent replay 用） |
| `Cream/` | 子模組：microsoft/Cream，使用其中的 `AutoFormer/`（ViT supernet 與 NAS） |
| `rockmate/` | 子模組：Rockmate，自動 rematerialization 工具 |
| `FLORA/` | 子模組：FLORA 低秩 supernet（目前**未使用**，保留作參考） |
| `flora_supernet_twcc/` | 在國網 Nano5 訓練 FLORA supernet 的腳本（目前**擱置**，改用 AutoFormer 官方權重） |

## 目前決策

- Backbone：AutoFormer 官方 **supernet-S**（embed 320/384/448、depth 12–14、MLP 3.0–4.0、heads 5–7），不自行在 ImageNet 上訓練 supernet。
- 資料集：CORe50 NICv2-391，輸入 128×128（65 tokens）。
- 模型與權重檔（`*.pth`）不要 commit 進 repo。
