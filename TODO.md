在 Raspberry Pi 4 上的持續物件學習

**目標**
在 Raspberry Pi 4（64 位元 OS）上，以 1 GB 訓練記憶體預算，對 CORe50 NICv2-391 串流做持續學習：每個新物件約 1 分鐘內學會、不遺忘舊物件。方法是 latent replay；backbone 和切層 $\ell$ 用 base paper（GECCO'25）的 **NAS + remat** 框架搜出來：從 AutoFormer supernet 做演化搜尋，每個候選照 Algorithm 1 評估。

**系統組成**
- **Backbone：NAS 從 AutoFormer supernet 搜出的子網路**
  - supernet：`weights/supernet-{tiny,small,base}.pth`（官方權重，不重新訓練，也不使用 FLORA），用 `Cream/AutoFormer` 的模型程式碼載入。
  - 選定的子網路匯出成獨立的 ViT，才部署到 Pi。不帶著整個 supernet，否則記憶體會把用不到的權重也算進去。
  - 參考點（也放進初始族群）：官方 AutoFormer-T / -S / -B 子網路（`Cream/AutoFormer/experiments/subnet/AutoFormer-*.yaml`；224 輸入下 75.3% / 81.7% / 82.4%，5.8M / 22.9M / 53.7M 參數）。
  - 對照組：DeiT-S、DeiT-B（timm），不參與搜尋。
  - 一律 128×128 輸入，65 tokens（絕對位置編碼從 14×14 內插成 8×8）。
- **切層：** 第 $1..\ell$ 層凍結，只做推論（`no_grad`）；第 $\ell+1..L$ 層加分類頭可訓練，FP32、SGD。$\ell$ 是 NAS 搜尋的變數之一。
- **Learn mode（每個新物件一次）：**
  1. 影格以 20 fps 進來，縮到 128×128，時間降採樣只留 20%。
  2. 留下的影格過凍結層，得到 latent。
  3. 新 latent 加上從 buffer 抽的 300 筆舊 latent，一起訓練上層：mini-batch 128、2 epochs。
  4. buffer 用 reservoir sampling 更新，容量 1500 筆，FP32 約 150 MB。
- **remat：** 用 Rockmate 的 CPU 版（`patches/rockmate-cpu.patch`），只在超出預算時才開。如果它在 Pi（aarch64）上跑不起來，退回用 `torch.utils.checkpoint` 對每個 block 做 checkpoint。
- **省電模式：** 只更新 SLDA 分類頭，不做 backprop。
- **Recognize mode：** 凍結層的前向結果可以和 Learn mode 共用。

**NAS：整合 remat 的演化搜尋（base paper 式 3 + Algorithm 1）**
- **搜尋空間：** tiny、small、base 三個 supernet 合併成一個候選池（embed 192–624，接近 base paper Table 3 的範圍）。
  - genome = (哪個 supernet, embed, depth, 每層的 mlp ratio 和 heads, $\ell$)，$\ell \in \{6, 9, 11\}$。
  - 交配和突變只在同一個 supernet 內做。
- **目標：** $\max \text{Acc} / T_{train}$，限制 $M \le 1$ GB。
  - $M$、$T_{train}$ 都是 Learn mode 的情境：只訓練第 $\ell+1..L$ 層，batch 128。
- **$T_{limit}$：** 同一個 $\ell$ 下，DeiT-S 只靠 remat 壓進 1 GB 時的每批時間（對應 base paper 的 Remat baseline）。
- **每個候選的評估（Algorithm 1）：** 用成本模型估計，不上 Pi。
  1. 估 $M$。
  2. $M$ > 1 GB 就套用 remat，重新估 $M$ 和 $T_{train}$。
  3. $T_{train}$ > $T_{limit}$ 就淘汰。
  4. 剩下的量代理準確率。
- **成本模型：** 照 base paper 的分工，在主機上搜尋、在 Pi 上部署。base paper 是在 Xeon 上搜尋、在 Pi 4 上微調（§4.3–4.4）；這裡更進一步，讓成本反映 Pi 4 的實況。
  - 記憶體：由子網路設定解析式計算權重、梯度、optimizer 狀態、activation，再用 Pi 實測的 peak RSS 校正常數項。
  - 時間：在 Pi 4 上量每種 block（embed × mlp ratio × heads）的前向和反向時間，做成查找表，候選的時間用加總估計。
  - remat 的額外時間：用 Pi 上實測的 Rockmate（或 checkpoint）倍率校正。
- **代理準確率：** 每個候選都在 CORe50 上完整跑 3 個 seed 不可能，改用短版 CORe50（少數 NICv2 batch、1 個 seed），在 Colab 或主機上跑。
  - 先用約 10 個候選驗證代理和完整結果的排名相關性（Kendall τ）。
- **流程：**
  1. 初始族群 = 官方 AutoFormer-T / -S / -B + 隨機子網路。
  2. 跑 N 代，每代取 top-k 交配、突變。
  3. 得到 Pareto 前緣。
- **最終驗證：** Pareto 前緣上前幾名（約 5 個）匯出成獨立模型，在 Pi 4 上用 cgroup 實測（含 Rockmate），並在 Colab 完整跑 NICv2-391 的 3 個 seed，選出最終設定。
- **對照組（證明 NAS 有用）：** 依 base paper 的定義：
  - Remat-only：DeiT-S 加 remat。
  - Compressed-only：搜尋空間裡最小的子網路。
  - 固定 backbone 列舉：DeiT-S / -B、AutoFormer-T / -S / -B，每個都搭配 $\ell$ = 6, 9, 11。

**量測規範**
- **1 GB 預算要真的限制住，不能只是量：** 用 `systemd-run --scope -p MemoryMax=1G` 包住訓練程序，同時記錄 peak RSS。
  - Raspberry Pi OS 可能預設沒開 memory cgroup：檢查 `/sys/fs/cgroup/cgroup.controllers`，沒有 `memory` 就在 `/boot/firmware/cmdline.txt` 加上 `cgroup_enable=memory cgroup_memory=1`。
- **時間：** 每批 wall-clock 時間，扣掉前 3 步的 warm-up。
- **散熱：** Pi 4 要裝散熱片和風扇，並記錄 `vcgencmd measure_temp` 和 `measure_clock arm`，有降頻的數據要標記出來。
- **評估指標：**
  - 準確度：final acc、avg acc。
  - 遺忘：每類的歷史最佳準確率減去最終準確率，再取平均。
  - 成本：每批時間、Learn mode 一次的總時間、peak RSS、buffer 大小。
  - 推論：inference fps。
  - NAS：成本模型的預測誤差、代理準確率的排名相關性。
  - stretch：能耗，用 USB 功率計量。
- **持續學習 baselines：** naive fine-tune、完整 latent replay、SLDA、FSA + SLDA。

**時程與交付物**

| 週 | 交付物 | 驗收標準 |
|---|---|---|
| W1–2（M1） | Pi 4 環境（64 位元 OS、PyTorch CPU、memory cgroup、Rockmate 實測）；任意子網路匯出成獨立 ViT；成本模型（block 時間查找表、記憶體公式，用約 10 個子網路在 Pi 上校正）；校正有效 GFLOPS；CIFAR-100 per-step 時間比例重現 [2] | Rockmate 在 ARM 上能用，或已切換到備案；成本模型在沒用來校正的子網路上，記憶體和時間誤差 ≤ 15% |
| W3–4（M2） | 代理準確率流程和排名相關性驗證；跑完演化 NAS、產出 Pareto 前緣；前幾名在 Pi 上實測並在 Colab 跑 3 個 seed；在 Pi 上端到端跑通 latent replay + remat | 選出的設定在 cgroup 1 GB 下跑得動，Learn mode 一次 ≤ 1 分鐘；Acc/T 優於固定 backbone 列舉的最佳者；比 SLDA 的差距 > 2σ |
| W5–6（M3） | Demo（Recognize → 新物件 → Learn → 再辨識，介面顯示時間和記憶體）；NICv2 的準確度和遺忘曲線；NAS 對照組的取捨圖（仿 base paper Fig. 4） | Demo 能現場跑；曲線和 Colab 的結果一致 |
| W7–8（stretch） | $\ell$ 的範圍擴大；INT8 buffer 和凍結層、latent 壓縮、能耗量測；如果拿到 FLORA supernet，換成低秩搜尋空間再跑一次 | MVP 完成後才開始 |

**近期工作順序（M1，先在本機做）**

| 順序 | 工作 | 產出 | 之後到 Pi 上要補的 |
|---|---|---|---|
| 1 | ~~**Rockmate CPU 可行性**~~ **已完成**：Rockmate 改成支援 CPU（子模組分支 `cpu-support`，patch 在 `patches/rockmate-cpu.patch`，測試在 `rockmate/test/test_cpu.py`），專案用的介面在 `utils/remat.py`（預算判斷、PureRotor / PureRockmate、checkpoint 備案），量測和預算換算在 `utils/perf.py`。Mac 上梯度和 PyTorch 一致；預算 = PyTorch 峰值的 50% 時，實測峰值 104 MiB（模擬 104 MiB，PyTorch 220 MiB）；時間倍率 1.3–1.4 倍 | remat 確定走 Rockmate | 在 Pi 上跑 `test_cpu.py`，實測時間倍率和求解時間 |
| 2 | **通用的子網路建構和匯出**：吃任意 genome（三個 supernet 都支援），匯出成獨立的 ViT，並驗證匯出前後的輸出一致；DeiT-S / -B 也套用同樣的 Frozen / Plastic 切法 | 第 1、3、4、5 項和 Pi 端都會用到 | 無 |
| 3 | **NAS 骨架**（`utils/nas.py`）：genome、交配和突變、Algorithm 1 的流程、fitness = Acc / T；成本模型和代理準確率先做成可替換的介面，暫時用假資料 | 整個搜尋流程能先跑通 | 換上真的查找表 |
| 4 | **記憶體公式**：從 genome 和 $\ell$ 算出權重、梯度、activation、buffer 的記憶體，在 Mac 上用 RSS 和 profiler 檢查結構是否正確 | 成本模型的記憶體部分 | 用 Pi 實測校正常數 |
| 5 | **持續學習核心**：下載 CORe50（128×128 版），寫 NICv2-391 串流、reservoir buffer、SLDA、評估指標，在 Mac（MPS）上跑小規模的 latent replay | 代理準確率的基礎、baselines 的初步數字 | 無 |
| 6 | **Pi 端量測腳本**：包好 `systemd-run`、量 peak RSS 和每批時間、記錄溫度和時脈，並產生 block 時間查找表 | 腳本寫好，到 Pi 上直接跑 | 在 Pi 上執行 |

第 1 項已完成；第 2 項是其他工作的基礎，也是 AutoFormer 能交給 Rockmate 的前提，接著做。

**Pi 4 上的第一輪測試（M1：環境和 Rockmate 驗證）**

目前只能測環境和 Rockmate；Learn mode 要等訓練迴圈、AutoFormer 匯出和量測腳本寫好。結果記在這份 TODO（或貼回來整理）。

- [ ] **0. 準備**
  - [ ] 確認 Pi 4 的記憶體大小（2 / 4 / 8 GB）。
  - [ ] 裝 64 位元 Raspberry Pi OS：`uname -m` 要是 `aarch64`。
  - [ ] 散熱片和風扇裝好。
  - [ ] 在 Mac 上把分支 push 上去：`git push -u origin feature/training-on-pi`。
- [ ] **1. 環境**
  - [ ] memory cgroup：`cat /sys/fs/cgroup/cgroup.controllers` 要有 `memory`。沒有的話在 `/boot/firmware/cmdline.txt` 加上 `cgroup_enable=memory cgroup_memory=1`，然後重開機。
  - [ ] `sudo apt install -y build-essential`（rotor 的 C 擴充要編譯）。
  - [ ] `git clone --recurse-submodules -b feature/training-on-pi <repo> && cd EAI-project`
  - [ ] 套用 Rockmate CPU patch：`git -C rockmate apply ../patches/rockmate-cpu.patch`。這一步會產生 `rockmate/test/test_cpu.py`；沒套用的話，後面的測試檔不存在，Rockmate 也還是原本只支援 CUDA 的版本。
  - [ ] `uv sync --extra remat`
  - [ ] PyTorch 能跑：`uv run python -c "import torch; x=torch.randn(512,512); print(torch.__version__, (x@x).sum())"`。出現 `Illegal instruction` 就要換較舊的 torch（Pi 4 是 ARMv8.0）。
  - [ ] 從 Mac 傳權重：`scp weights/*.pth <pi>:~/EAI-project/weights/`
- [ ] **2. Rockmate 正確性**：`uv run python -m pytest rockmate/test/test_cpu.py -s`
  - [ ] 5 項測試全過（梯度和 PyTorch 一致、峰值壓在預算內）。
  - [ ] 記下每項測試的時間（主要是求解時間）。
  - [ ] PureRockmate 需要 CBC 求解器；pulp 內建的版本不一定有 ARM 版，缺的話 `sudo apt install coinor-cbc`。預設的 PureRotor 不需要。
- [ ] **3. remat 實測**：`uv run python -m utils.remat --budget-ratio 0.5`
  - [ ] 記下：原本的峰值和每步時間、remat 後的峰值和每步時間、時間倍率、求解時間。
  - [ ] 加上 1 GB 限制再跑一次，必須關掉 swap，否則超出的部分會被換到 swap，等於沒限制：
    `sudo systemd-run --scope -p MemoryMax=1G -p MemorySwapMax=0 --uid=$USER uv run python -m utils.remat --budget-ratio 0.5`
  - [ ] 跑的時候另開終端機記錄溫度和時脈：`watch -n 2 'vcgencmd measure_temp; vcgencmd measure_clock arm'`，有降頻要標記。
- [ ] **4. Python + PyTorch 本身的記憶體**（`perf.step_budget()` 的 `runtime_bytes`）：
  `uv run python -c "import torch, psutil; print(psutil.Process().memory_info().rss/2**20, 'MiB')"`
- [ ] **5. AutoFormer-S 實測**：`uv run python -m utils.subnet_split -l 9 --batch 32`，再試 `-l 6`、`-l 11`。
  - [ ] 記下凍結層的推論時間、後段一次訓練步的時間。
  - [ ] 和 Mac 的數字比較，得到 Pi 4 的速度倍率，用來修正「Pi 4 的速度」那項風險的粗估。

**現有程式碼**
- `utils/subnet_split.py`：載入 supernet-S（128×128）並切成 Frozen / Plastic，目前只寫死 AutoFormer-S。
- `utils/nas.py`：空殼，演化搜尋要寫在這裡。可參考 `Cream/AutoFormer/evolution.py`（原本的限制是參數量，要換成 Algorithm 1）。
- 要新增的部分：
  - 持續學習核心：CORe50 NICv2-391 串流、reservoir buffer、SLDA、評估指標（final / avg acc、遺忘）。
  - Colab 實驗：latent replay 各變體和 baselines（naive fine-tune、SLDA、FSA + SLDA），以及 NAS 用的代理準確率。
  - 通用的子網路建構和匯出：吃任意 genome，三個 supernet 都支援，匯出成獨立 ViT。DeiT-S / -B 也用同樣的 Frozen / Plastic 切法。
  - Pi 端量測腳本：單一設定的 peak RSS 和每批時間，加上 block 時間查找表。
  - 成本模型：記憶體公式、查找表、remat 倍率。
  - 代理準確率評估。
  - Pi 端的訓練迴圈：把 cached latent 換成即時的前向計算。
  - Demo 介面。

**已知風險與未決事項**
- **Pi 4 的速度：** 比 Pi 5 慢約 2–3 倍。base paper Table 4 換算，Pi 4 完整微調 DeiT-S 約每秒 1 張；粗估 Learn mode 在 $\ell$ = 6 要約 2 分鐘，$\ell$ = 9 約 1 分鐘。1 分鐘的目標可能要靠較大的 $\ell$ 或較小的子網路才達得到，這也是 NAS 要解決的事。
- **Pi 4 的記憶體大小：** 待確認。2 GB 版本和 base paper 相同；4 或 8 GB 版本要靠 cgroup 硬性限制。
- **成本模型的準確度：** block 加總會忽略 runtime 開銷和快取效應，所以最終候選一定要上 Pi 實測。
- **代理準確率的可靠度：** 排名相關性太低的話，要加大代理的規模（更多 batch），或改用 ImageNet val 子集。
- **Rockmate 的 CPU 版：** 已在 Mac 上驗證（見近期工作第 1 項），尚未在 Pi（aarch64）上跑過。
  - 修改只在子模組的工作目錄裡（分支 `cpu-support`，未 commit）。別人 clone 或在 Pi 上要先 `git -C rockmate apply ../patches/rockmate-cpu.patch`，或改用 fork。
  - 只支援不做 offload 的 solver（`PureRockmate`、`PureCheckmate`、`Hiremate`）；`Offmate` 在 CPU 上沒有意義，會直接報錯。
  - 求解時間（Mac，2 個 DeiT-S block、batch 64）：`PureRotor` 約 20 秒、`PureRockmate`（ILP）約 4 分鐘，Pi 4 會更久。`utils/remat.py` 預設用 `PureRotor`。NAS 只對少數最終候選跑 Rockmate；必要時用 `save_sched_to_local` 把排程存起來重複使用。
- **AutoFormer 不能直接交給 Rockmate（已確認）：** super 模組在 `set_sample_config` 時把切好的權重存成一般屬性，`torch.export` 把它們當成常數抽出去，梯度也不會流回參數。必須先匯出成獨立的 ViT（近期工作第 2 項）。
- **AutoFormer 在 128×128 的準確率：** 官方數字是 224 輸入，位置編碼內插後會掉多少未知，要先抽 ImageNet val 的一部分驗證。
- **supernet 的搜尋範圍：** 官方 supernet 的範圍比 base paper 窄（論文 Table 3 是自己擴大範圍、訓練 700 epochs），取捨曲線會比較短，用三個 supernet 合併補救。
- **和 base paper 的數字無法直接比：** 論文的 Pi 實驗（Table 4、5）用的是 FLORA DeiT-S 和 CIFAR-100 / CUB-200。所有對照組都要在同一台 Pi、同一套設定下自己重跑。
- **記憶體數字：** 目前全是解析式估算，不含 runtime，M1 要用實測校正。
- **標籤來源：** 假設是有標籤的 session，demo 用 CORe50 的標籤模擬。
