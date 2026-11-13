"""
從 AutoFormer 官方 supernet-small 切出子網路，並在第 l 層切成
  Frozen  (patch embed + block 1..l，凍結、只推論，輸出 latent)
  Plastic (block l+1..L + norm + 50 類 head，在裝置上訓練)
給 latent replay 持續學習使用。

用法（在 EAI-project 根目錄執行）：
  python autoformer_cl/subnet_split.py --ckpt ../weights/supernet-small.pth --img-size 128 --latent-layer 9

需要：torch、numpy（AutoFormer 的模型檔只依賴這兩個）
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Cream", "AutoFormer"))

# AutoFormer 是為 torch 1.7 寫的，會 import 已在 torch 2.x 移除的 torch._six。
# 這裡補一個相容模組，不必修改 Cream 子模組本身。
if "torch._six" not in sys.modules:
    import collections.abc
    import types
    _six = types.ModuleType("torch._six")
    _six.container_abcs = collections.abc
    _six.string_classes = (str, bytes)
    _six.int_classes = int
    sys.modules["torch._six"] = _six
from model.supernet_transformer import Vision_TransformerSuper  # noqa: E402
from model.module.Linear_super import LinearSuper                # noqa: E402

# supernet-S 的最大架構（Cream/AutoFormer/experiments/supernet/supernet-S.yaml）
SUPERNET_S = dict(embed_dim=448, depth=14, num_heads=7, mlp_ratio=4.0)

# 官方搜尋出的 AutoFormer-S 子網路（Cream/AutoFormer/experiments/subnet/AutoFormer-S.yaml，ImageNet 81.7%）
AUTOFORMER_S = dict(
    embed_dim=[384] * 13,
    mlp_ratio=[3.0, 3.5, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.5, 4.0],
    num_heads=[6, 6, 5, 7, 5, 5, 5, 6, 6, 7, 7, 6, 7],
    layer_num=13,
)


def build_supernet(ckpt_path, img_size=128, num_classes=1000):
    # 官方測試指令使用 --gp --change_qk --relative_position，這裡一併開啟
    model = Vision_TransformerSuper(
        img_size=img_size, patch_size=16, num_classes=1000, qkv_bias=True,
        gp=True, change_qkv=True, relative_position=True, max_relative_position=14,
        **SUPERNET_S)

    # 權重是在 GPU 上存的，一定要 map_location='cpu'（Pi 沒有 CUDA）
    state = torch.load(ckpt_path, map_location="cpu")["model"]

    # 絕對位置編碼：224 輸入是 1+14×14=197 個 token，改成 img_size 時要內插
    pe = state["pos_embed"]                                   # (1, 197, 448)
    n_old = int((pe.shape[1] - 1) ** 0.5)
    n_new = img_size // 16
    if n_new != n_old:
        cls_pe, grid = pe[:, :1], pe[:, 1:]
        grid = grid.reshape(1, n_old, n_old, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(n_new, n_new), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, n_new * n_new, -1)
        state["pos_embed"] = torch.cat([cls_pe, grid], dim=1)
    # 相對位置編碼依序列長度動態計算，不需要調整

    msg = model.load_state_dict(state, strict=True)
    print("load_state_dict:", msg)

    # 換成 CORe50 的 50 類 head（ImageNet 的 1000 類 head 用不到）
    if num_classes != 1000:
        model.num_classes = num_classes
        model.head = LinearSuper(SUPERNET_S["embed_dim"], num_classes)
    return model


class Frozen(nn.Module):
    """patch embedding + block 1..l，輸出第 l 層的 latent：(B, 1+N, D)"""

    def __init__(self, sup, l):
        super().__init__()
        self.sup, self.l = sup, l

    @torch.no_grad()
    def forward(self, x):
        s = self.sup
        d = s.sample_embed_dim[0]
        x = s.patch_embed_super(x)
        x = torch.cat([s.cls_token[..., :d].expand(x.shape[0], -1, -1), x], dim=1)
        x = x + s.pos_embed[..., :d]
        for blk in s.blocks[: self.l]:
            x = blk(x)
        return x


class Plastic(nn.Module):
    """block l+1..L + LayerNorm + global pooling + head"""

    def __init__(self, sup, l):
        super().__init__()
        self.sup, self.l = sup, l

    def forward(self, z):
        s = self.sup
        for blk in s.blocks[self.l:]:       # 超過 layer_num 的 block 是 identity
            z = blk(z)
        z = s.norm(z)
        z = z[:, 1:].mean(dim=1) if s.gp else z[:, 0]
        return s.head(z)


def trainable_parameters(sup, l):
    """只回傳後段（block l+1..L、norm、head）的參數，交給 optimizer"""
    mods = list(sup.blocks[l:]) + [sup.norm, sup.head]
    return [p for m in mods for p in m.parameters()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "..", "weights", "supernet-small.pth"))
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--latent-layer", "-l", type=int, default=9)
    ap.add_argument("--num-classes", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    sup = build_supernet(args.ckpt, args.img_size, args.num_classes)

    # 注意：set_sample_config 會先切好權重的 view；
    # 若之後要 .to(device)，請在 .to() 之後再呼叫一次 set_sample_config
    cfg = AUTOFORMER_S
    sup.set_sample_config(cfg)
    n_params = sup.get_sampled_params_numel(cfg)
    print(f"子網路參數量：{n_params / 1e6:.2f}M（含 {args.num_classes} 類 head）")

    l = args.latent_layer
    assert 0 < l < cfg["layer_num"], "latent layer 要介於 1 和 layer_num-1 之間"
    frozen, plastic = Frozen(sup, l), Plastic(sup, l)

    # 凍結前段：關掉梯度並切到 eval（關閉 dropout / drop path）
    for m in [sup.patch_embed_super] + list(sup.blocks[:l]):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    sup.pos_embed.requires_grad_(False)
    sup.cls_token.requires_grad_(False)

    x = torch.randn(args.batch, 3, args.img_size, args.img_size)
    t0 = time.time()
    z = frozen(x)
    t1 = time.time()
    print(f"latent 形狀：{tuple(z.shape)}，前段推論 {1000 * (t1 - t0):.0f} ms / {args.batch} 張")
    print(f"每筆 latent 大小（FP32）：{z[0].numel() * 4 / 1024:.1f} KB；1500 筆 buffer ≈ {1500 * z[0].numel() * 4 / 2**20:.1f} MB")

    params = trainable_parameters(sup, l)
    print(f"可訓練參數（含未取樣到的 supernet 部分）：{sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9)
    y = torch.randint(0, args.num_classes, (args.batch,))
    t0 = time.time()
    # AutoFormer 在 set_sample_config 時才把權重切成 view；
    # 每次訓練步前重新呼叫一次（很便宜），確保用到 optimizer 更新後的權重並重建計算圖
    sup.set_sample_config(cfg)
    loss = F.cross_entropy(plastic(z), y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"後段一次訓練步 {1000 * (time.time() - t0):.0f} ms，loss = {loss.item():.3f}")


if __name__ == "__main__":
    main()
