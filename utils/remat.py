"""
remat.py：在 CPU（Mac / Raspberry Pi）上依記憶體預算對訓練中的模型做 rematerialization。

對應 base paper Algorithm 1 第 9–13 行：候選子網路的訓練記憶體超過預算時，
用 remat 壓進預算，再量 remat 之後的訓練時間。

做法（apply_remat）：
  1. 先量原模型一次訓練步（forward + backward）的峰值記憶體，沒超過預算就不開 remat。
  2. 超過就用 Rockmate（CPU 版，見 patches/rockmate-cpu.patch）求出 remat 排程。
     solver="rotor"（預設）用 PureRotor 的動態規劃，求解快；
     solver="rockmate" 用 PureRockmate（ILP），慢很多（Mac 上 2 個 block 就要約 4 分鐘）。
  3. Rockmate 失敗或找不到可行排程時，退回 torch.utils.checkpoint：
     由前往後逐個 block 開 checkpoint，直到峰值記憶體進入預算。

「預算」的定義：一次訓練步在「已常駐的記憶體」之外額外配置的峰值。
已常駐的部分是權重、梯度、optimizer 狀態、replay buffer 和 Python/PyTorch 本身，
不在這個預算裡。量測和預算換算在 perf.py（measure_step、step_budget）。

注意：
  - 模型必須是一般的 nn.Module。AutoFormer 的 super 模組會在 set_sample_config 時
    把切好的權重存成一般屬性，torch.export 會把它們當常數，梯度不會流回參數；
    要先把子網路匯出成獨立的 ViT 再交給 Rockmate。
  - 求解時間（Mac，2 個 DeiT-S block、batch 64）：PureRotor 約 20 秒，PureRockmate 約 4 分鐘；
    Pi 4 會更久。

用法：
  from utils.perf import step_budget, GiB
  from utils.remat import apply_remat
  budget = step_budget(1 * GiB, plastic, optimizer, extra_bytes=buffer_bytes, runtime_bytes=...)
  res = apply_remat(plastic, [z], budget)
  y = res.model(z)            # 和原模型一樣的輸出，照常 loss.backward()、optimizer.step()

示範（在 EAI-project 根目錄執行）：
  uv run python -m utils.remat --budget-ratio 0.5
"""

import argparse
import contextlib
import io
import time
import types
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from utils.perf import MiB, StepStats, default_loss, measure_step


# =====================================================================
# 備案：block 層級的 torch.utils.checkpoint
# =====================================================================


def find_blocks(model: nn.Module, attr: Optional[str] = None) -> Sequence[nn.Module]:
    """找出要做 checkpoint 的 block 序列：預設找名為 blocks 的 Sequential / ModuleList"""
    if attr is not None:
        blocks = model.get_submodule(attr)
    elif hasattr(model, "blocks"):
        blocks = model.blocks
    else:
        raise ValueError("找不到 model.blocks，請用 blocks_attr 指定 block 序列的名稱")
    if not isinstance(blocks, (nn.Sequential, nn.ModuleList)):
        raise ValueError(
            f"block 序列必須是 nn.Sequential 或 nn.ModuleList，而不是 {type(blocks)}"
        )
    return list(blocks)


def _checkpointed_forward(self, *args, **kwargs):
    # 不經過 __call__，直接呼叫類別的 forward，避免遞迴
    fwd = type(self).forward.__get__(self)
    if not torch.is_grad_enabled():
        return fwd(*args, **kwargs)
    return checkpoint(fwd, *args, use_reentrant=False, **kwargs)


def set_checkpoint(blocks: Sequence[nn.Module], k: int):
    """前 k 個 block 開 checkpoint，其餘關閉。直接改實例的 forward，參數名稱不變"""
    for i, blk in enumerate(blocks):
        if i < k:
            blk.forward = types.MethodType(_checkpointed_forward, blk)
        elif "forward" in blk.__dict__:
            del blk.forward


# =====================================================================
# 主介面
# =====================================================================


class _Untuple(nn.Module):
    """讓 Rockmate 模組的輸出和原模型一致（Rockmate 一律回傳 tuple）"""

    def __init__(self, rk):
        super().__init__()
        self.rk = rk

    def forward(self, *args, **kwargs):
        y = self.rk(*args, **kwargs)
        return y[0] if isinstance(y, tuple) and len(y) == 1 else y

    def zero_grad(self, set_to_none: bool = False):
        self.rk.zero_grad(set_to_none=set_to_none)


@dataclass
class RematResult:
    model: nn.Module  # 拿來訓練的模組，輸出和原模型相同
    method: str  # "none" | "rockmate" | "checkpoint"
    budget: int
    baseline: StepStats  # 不做 remat 的量測
    stats: StepStats  # 套用 remat 之後的量測
    fits: bool  # stats.peak_bytes <= budget
    checkpointed_blocks: int = 0  # method == "checkpoint" 時開了幾個 block
    simulated_peak: Optional[int] = None  # Rockmate 自己模擬的峰值
    notes: list = field(default_factory=list)

    @property
    def time_ratio(self) -> float:
        """remat 之後的時間 / 原本的時間：成本模型要用的 remat 倍率"""
        return self.stats.step_ms / self.baseline.step_ms

    def __str__(self):
        s = (
            f"method={self.method} | budget {self.budget / MiB:.1f} MiB | "
            f"baseline {self.baseline} | remat {self.stats} | "
            f"time x{self.time_ratio:.2f} | fits={self.fits}"
        )
        if self.method == "checkpoint":
            s += f" | checkpointed blocks={self.checkpointed_blocks}"
        if self.simulated_peak is not None:
            s += f" | Rockmate simulated {self.simulated_peak / MiB:.1f} MiB"
        return s


SOLVERS = ("rotor", "rockmate")


def _rockmate(model, inputs, budget, quiet, solver):
    from rockmate import PureRockmate, PureRotor

    cls = {"rotor": PureRotor, "rockmate": PureRockmate}[solver]
    out = io.StringIO()
    ctx = contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext()
    with ctx:
        rk = cls(model, list(inputs), budget=budget, device="cpu")
    if rk.op_sched is None:
        raise RuntimeError("Rockmate 找不到符合預算的排程")
    return rk


def apply_remat(
    model: nn.Module,
    inputs: Sequence[torch.Tensor],
    budget: int,
    method: str = "auto",
    solver: str = "rotor",
    loss_fn: Callable = default_loss,
    blocks_attr: Optional[str] = None,
    quiet: bool = True,
    iters: int = 5,
) -> RematResult:
    """
    讓 model 的一次訓練步（在 inputs 上）壓進 budget bytes。

    method：
      "auto"       沒超過預算就不做；超過先試 Rockmate，失敗再退回 checkpoint
      "rockmate"   只用 Rockmate
      "checkpoint" 只用 block 層級的 torch.utils.checkpoint
      "none"       不做 remat，只量測
    solver："rotor"（PureRotor，預設，快）或 "rockmate"（PureRockmate，ILP，慢）。
    inputs 要和實際訓練時的形狀（尤其 batch size）相同，Rockmate 的排程是依這個形狀求的。
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver 要是 {SOLVERS} 之一")
    if any(t.is_cuda for t in inputs) or any(p.is_cuda for p in model.parameters()):
        raise ValueError("remat.py 是給 CPU 用的；模型和輸入都要在 CPU 上")
    model.train()
    inputs = list(inputs)
    baseline = measure_step(model, inputs, loss_fn, iters=iters)

    def result(m, name, stats, **kw):
        return RematResult(
            m, name, budget, baseline, stats, stats.peak_bytes <= budget, **kw
        )

    if method == "none" or (method == "auto" and baseline.peak_bytes <= budget):
        return result(model, "none", baseline)

    notes = []
    if method in ("auto", "rockmate"):
        try:
            rk = _rockmate(model, inputs, budget, quiet, solver)
            wrapped = _Untuple(rk)
            stats = measure_step(wrapped, inputs, loss_fn, iters=iters)
            res = result(
                wrapped, "rockmate", stats, simulated_peak=int(rk.op_sched.peak_mem)
            )
            res.notes.append(f"solver={solver}")
            if res.fits or method == "rockmate":
                return res
            notes.append(f"Rockmate 實測峰值 {stats.peak_bytes / MiB:.1f} MiB 超過預算")
        except Exception as e:  # noqa: BLE001  Rockmate 失敗就退回 checkpoint
            if method == "rockmate":
                raise
            notes.append(f"Rockmate 失敗：{type(e).__name__}: {e}")

    # 備案：由前往後逐個 block 開 checkpoint，直到進入預算
    blocks = find_blocks(model, blocks_attr)
    stats = baseline
    k = 0
    for k in range(1, len(blocks) + 1):
        set_checkpoint(blocks, k)
        stats = measure_step(model, inputs, loss_fn, iters=iters)
        if stats.peak_bytes <= budget:
            break
    return result(model, "checkpoint", stats, checkpointed_blocks=k, notes=notes)


# =====================================================================
# 示範：DeiT 結構的後段（Plastic），模擬 Learn mode 的一次訓練步
# =====================================================================


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_ratio * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_ratio * dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        h = self.heads
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
        attn = (qkv[0] @ qkv[1].transpose(-2, -1)) * (C // h) ** -0.5
        y = (attn.softmax(dim=-1) @ qkv[2]).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(y)
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class DemoPlastic(nn.Module):
    """第 l+1..L 層 + LayerNorm + 平均池化 + 分類頭"""

    def __init__(self, depth=4, dim=384, heads=6, classes=50):
        super().__init__()
        self.blocks = nn.Sequential(*[_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, z):
        return self.head(self.norm(self.blocks(z))[:, 1:].mean(dim=1))


def main():
    ap = argparse.ArgumentParser(
        description="remat 示範：DeiT-S 結構的後段，128×128 輸入（65 tokens）"
    )
    ap.add_argument("--depth", type=int, default=4, help="要訓練的 block 數（L - l）")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=65)
    ap.add_argument(
        "--budget-ratio", type=float, default=0.5, help="預算 = 原本峰值 × 這個比例"
    )
    ap.add_argument(
        "--method", default="auto", choices=["auto", "rockmate", "checkpoint", "none"]
    )
    ap.add_argument("--solver", default="rotor", choices=list(SOLVERS))
    ap.add_argument("--verbose", action="store_true", help="顯示 Rockmate 求解過程")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = DemoPlastic(args.depth, args.dim, args.heads)
    z = torch.randn(args.batch, args.tokens, args.dim)
    base = measure_step(model, [z])
    budget = int(base.peak_bytes * args.budget_ratio)
    print(f"原本：{base}；預算 {budget / MiB:.1f} MiB")
    t0 = time.time()
    res = apply_remat(
        model,
        [z],
        budget,
        method=args.method,
        solver=args.solver,
        quiet=not args.verbose,
    )
    print(f"套用 remat 耗時 {time.time() - t0:.0f} s")
    print(res)
    for n in res.notes:
        print("  -", n)


if __name__ == "__main__":
    main()
