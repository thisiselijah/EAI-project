"""
utils：供 src/ 和實驗腳本共用的工具（在 repo 根目錄用 `python -m utils.<模組>` 執行）。

  perf          訓練的記憶體與時間量測、記憶體預算換算
  remat         依記憶體預算做 rematerialization（Rockmate CPU 版 + checkpoint 備案）
  subnet_split  從 AutoFormer supernet 切出子網路，分成 Frozen / Plastic
  nas           演化搜尋（尚未實作）
"""
