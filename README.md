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
uv run python -m utils.subnet_split -l 9
```

第一次 `uv sync` 會產生 `uv.lock`，請一起 commit。

### 下載預訓練權重（不放在 git 裡）

把權重放在 repo 裡的 `weights/`（已在 `.gitignore`，不會被 commit）：

```bash
mkdir -p weights && cd weights
# AutoFormer 官方 supernet-S（139 MB）
wget https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-small.pth
# 官方 DeiT-S（88 MB，選用）
wget https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
cd ..
```

### 測試：切出子網路並在第 l 層分段

```bash
uv run python -m utils.subnet_split -l 9   # l = latent layer，可試 6 / 9 / 11
```

## 專案結構

| 路徑 | 內容 |
|---|---|
| `src/` | 在 Pi 上執行的主程式：訓練（Learn mode）、辨識（Recognize mode）。用 `python -m src.<模組>` 執行 |
| `utils/` | 共用工具：子網路切分（`subnet_split`）、量測（`perf`）、remat（`remat`）、NAS（`nas`）。用 `python -m utils.<模組>` 執行 |
| `Cream/` | 子模組：microsoft/Cream，使用其中的 `AutoFormer/`（ViT supernet 與 NAS） |
| `rockmate/` | 子模組：Rockmate，自動 rematerialization 工具 |
| `FLORA/` | 子模組：FLORA 低秩 supernet（目前**未使用**，保留作參考） |
| `flora_supernet_twcc/` | 在國網 Nano5 訓練 FLORA supernet 的腳本（目前**擱置**，改用 AutoFormer 官方權重） |




