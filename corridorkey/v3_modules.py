"""CorridorKey 3.0 additions to stage 2 (all zero-initialized: at insertion
the model reproduces 2.0 plus the 2.0 inference fixes exactly).

TemporalTokenMix   per-token attention across the frames of a window at /16,
                   so fine detail stops being decided frame by frame.
KeyTail            full-resolution output tail, run in fp32:
                     * EdgeRefiner: a small /1 conv net over raw pixels, the
                       current alpha/foreground, upsampled stage-2 tokens and
                       a screen-measured noise level; predicts residuals.
                     * Foreground fusion: plate-recovered foreground
                       (I - (1 - a) * B_hat) / a, despilled against B_hat, is
                       blended with the network foreground. A /16 head learns
                       per-pixel corrections to the spill amount and to the
                       blend weight, around the analytic 2.0-fix values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import color_utils as cu
from .blocks import RMSNorm
from .plate_fg import _erode, pull_push


class TemporalTokenMix(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, max_dt: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_dt = max_dt
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.rel_bias = nn.Parameter(torch.zeros(2 * max_dt + 1, num_heads))
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """x: (B*T, h, w, C) -> same shape."""
        if T <= 1:
            return x
        BT, h, w, C = x.shape
        y = self.norm(x).view(B, T, h * w, C).permute(0, 2, 1, 3).reshape(B * h * w, T, C)
        q, k, v = self.qkv(y).view(-1, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        dt = torch.arange(T, device=x.device)
        dt = (dt[None, :] - dt[:, None]).clamp(-self.max_dt, self.max_dt) + self.max_dt
        bias = self.rel_bias[dt].permute(2, 0, 1).to(q.dtype)          # (heads, T, T)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)     # (N, heads, T, hd)
        o = o.permute(0, 2, 1, 3).reshape(B, h * w, T, C).permute(0, 2, 1, 3).reshape(BT, h, w, C)
        return x + self.proj(o)


class EdgeRefiner(nn.Module):
    def __init__(self, token_dim: int, width: int = 32, token_ch: int = 8, depth: int = 3):
        super().__init__()
        self.token_proj = nn.Conv2d(token_dim, token_ch, 1)
        in_ch = 3 + 1 + 3 + 1 + token_ch
        self.inp = nn.Conv2d(in_ch, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                          nn.Conv2d(width, width, 3, padding=1))
            for _ in range(depth)])
        self.out = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, rgb, alpha, fg, log_sigma, tokens):
        H, W = rgb.shape[-2:]
        t = F.interpolate(self.token_proj(tokens), size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([rgb, alpha, fg, log_sigma.expand(-1, 1, H, W), t], dim=1)
        x = F.gelu(self.inp(x))
        for blk in self.blocks:
            x = x + blk(x)
        return self.out(x)


def _logit(p, eps=1e-3):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


def _smoothstep(lo, hi, x):
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def screen_noise(plate_lin, alpha):
    """Per-frame noise level measured on confident screen pixels. (N,1,1,1)"""
    bg = (alpha < 0.01).float()
    resid = (plate_lin - F.avg_pool2d(plate_lin, 5, 1, 2)).abs().mean(1, keepdim=True)
    s = (resid * bg).sum((-2, -1), keepdim=True) / bg.sum((-2, -1), keepdim=True).clamp(min=1.0)
    return torch.log(s.clamp(min=1e-5))


def analytic_spill_k(fg, backing):
    """Despill coefficient k with F - k * B removing screen-coloured light (see plate_fg)."""
    dom = backing.mean(dim=(-2, -1), keepdim=True).argmax(dim=1, keepdim=True)
    dom = dom.expand(-1, 1, *fg.shape[-2:])
    o1, o2 = (dom + 1) % 3, (dom + 2) % 3
    f_d, b_d = fg.gather(1, dom), backing.gather(1, dom)
    f_o = torch.maximum(fg.gather(1, o1), fg.gather(1, o2))
    b_o = torch.maximum(backing.gather(1, o1), backing.gather(1, o2))
    return (f_d - f_o).clamp(min=0) / (b_d - b_o).clamp(min=1e-4)


class KeyTail(nn.Module):
    def __init__(self, token_dim: int, refiner_width: int = 32):
        super().__init__()
        self.refiner = EdgeRefiner(token_dim, width=refiner_width)
        self.fusion = nn.Conv2d(token_dim, 2 * 16, 1)     # (spill, blend) at /4 via pixel shuffle
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, rgb_asinh, logits, tokens, refine_band=None):
        """rgb_asinh (N,3,H,W); logits (N,4,H,W) alpha + asinh FG; tokens (N,d,h,w).

        Returns final logits (N,4,H,W) with alpha and asinh-encoded fused FG, and
        a dict of intermediates for losses.
        """
        rgb_asinh = rgb_asinh.float()
        logits = logits.float()
        tokens = tokens.float()
        plate = cu.asinh_to_linear(rgb_asinh)
        a0 = logits[:, 0:1]
        f0 = logits[:, 1:4]
        log_sigma = screen_noise(plate, a0.detach().clamp(0, 1))
        delta = self.refiner(rgb_asinh, a0, f0, log_sigma, tokens)
        if refine_band is not None:
            delta = delta * refine_band
        alpha = a0 + delta[:, 0:1]
        f_net = cu.asinh_to_linear(f0 + delta[:, 1:4])

        a = alpha.detach().clamp(0, 1)
        with torch.no_grad():
            backing = pull_push(plate, _erode((a < 0.004).float(), 2))
        H, W = plate.shape[-2:]
        heads = F.pixel_shuffle(self.fusion(tokens), 4)
        heads = F.interpolate(heads, size=(H, W), mode="bilinear", align_corners=False)
        a_safe = a.clamp(min=0.02)
        f_plate = (plate - (1 - a) * backing) / a_safe
        k = analytic_spill_k(f_plate, backing) * (1 + torch.tanh(heads[:, 0:1]))
        f_plate = f_plate - k * backing
        w = torch.sigmoid(_logit(_smoothstep(0.05, 0.6, a)) + heads[:, 1:2])
        w = w * (a > 0.02).float()
        f_final = w * f_plate + (1 - w) * f_net
        out = torch.cat([alpha, cu.linear_to_asinh(f_final)], dim=1)
        return out, {"f_net": f_net, "f_plate": f_plate, "w": w, "k": k, "backing": backing}
