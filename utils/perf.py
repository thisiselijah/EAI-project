"""
perf.py：在 CPU（Mac / Raspberry Pi）上量訓練的記憶體和時間，並做記憶體預算換算。

給 remat.py、NAS 的成本模型和 Pi 端量測腳本共用：`from utils.perf import measure_step, step_budget`

  measure_step()   一次訓練步（forward + backward）的峰值記憶體和每步時間
  peak_memory()    任意函式執行期間的淨增配置和峰值配置（PyTorch profiler 的記憶體事件）
  resident_bytes() 訓練時常駐的模型記憶體：參數 + 梯度 + optimizer 狀態
  step_budget()    把總預算（例如 1 GB）換算成一次訓練步可用的預算

量測慣例：warm-up 之後梯度常駐（zero_grad(set_to_none=False)），
所以 measure_step 的峰值只包含 activation 和暫存空間。實際訓練迴圈也請照這個方式。
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

MiB = 1024**2
GiB = 1024**3


# =====================================================================
# 量測
# =====================================================================


def _as_tensor_output(y):
    """Rockmate 會把輸出包成 tuple；統一取出第一個輸出"""
    return y[0] if isinstance(y, (tuple, list)) else y


def default_loss(y):
    """只用來量記憶體和時間的假 loss；真的訓練請傳入自己的 loss_fn"""
    return _as_tensor_output(y).float().square().mean()


def train_step(model, inputs: Sequence[torch.Tensor], loss_fn: Callable = default_loss):
    loss = loss_fn(model(*inputs))
    loss.backward()
    return loss


def peak_memory(fn: Callable, *args):
    """
    回傳 (fn 的結果, 淨增配置 bytes, 峰值配置 bytes)。

    用 PyTorch profiler 的記憶體事件精準量 CPU allocator 的配置量
    （rkgb 的 MemoryTrackerCPU），只算 fn 執行期間新配置的部分。
    """
    from rkgb.lowlevel.measure import MemoryTrackerCPU

    return MemoryTrackerCPU().measure(fn, *args)


@dataclass
class StepStats:
    peak_bytes: int  # 一次訓練步額外配置的峰值
    step_ms: float  # 每步 wall-clock 時間（不含 warm-up）

    def __str__(self):
        return f"peak {self.peak_bytes / MiB:.1f} MiB, {self.step_ms:.1f} ms/step"


def measure_step(
    model,
    inputs: Sequence[torch.Tensor],
    loss_fn: Callable = default_loss,
    iters: int = 5,
    warmup: int = 2,
) -> StepStats:
    """
    量一次訓練步的峰值記憶體和時間。
    warm-up 之後梯度已經配置好並常駐（zero_grad(set_to_none=False)），
    所以峰值只包含 activation 和暫存空間。
    計時的迭代不開 profiler，避免 profiler 本身的開銷算進時間。
    """

    def run():
        train_step(model, inputs, loss_fn)

    for _ in range(warmup):
        run()
        model.zero_grad(set_to_none=False)
    _, _, peak = peak_memory(run)
    model.zero_grad(set_to_none=False)
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
        model.zero_grad(set_to_none=False)
    return StepStats(peak, (time.perf_counter() - t0) / iters * 1000)


# =====================================================================
# 預算換算
# =====================================================================


def tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def resident_bytes(
    model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """
    訓練時常駐的模型記憶體：所有參數 + 可訓練參數的梯度 + optimizer 狀態。
    optimizer 狀態要在第一次 step() 之後才存在；之前就用參數量估計。
    """
    params = list(model.parameters())
    total = tensor_bytes(params)
    total += tensor_bytes(p for p in params if p.requires_grad)  # 梯度
    if optimizer is not None:
        total += tensor_bytes(
            v
            for state in optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor) and v.dim() > 0
        )
    return total


def step_budget(
    total_bytes: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra_bytes: int = 0,
    runtime_bytes: int = 0,
) -> int:
    """
    把總記憶體預算（例如 1 GB 的 cgroup 上限）換算成一次訓練步可用的預算。

    extra_bytes：其他常駐資料，例如 replay buffer、凍結層的權重（若不在 model 裡）。
    runtime_bytes：Python 和 PyTorch 本身的 RSS（import 完、還沒建模型時的 RSS），
                   要在 Pi 上實測，這裡不預設數值。
    """
    budget = (
        total_bytes - resident_bytes(model, optimizer) - extra_bytes - runtime_bytes
    )
    if budget <= 0:
        raise ValueError(
            f"常駐記憶體已超過總預算：總預算 {total_bytes / MiB:.0f} MiB，"
            f"模型 {resident_bytes(model, optimizer) / MiB:.0f} MiB，"
            f"其他 {(extra_bytes + runtime_bytes) / MiB:.0f} MiB"
        )
    return budget
