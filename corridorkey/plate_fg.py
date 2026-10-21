"""Recover the foreground from the plate itself ("screen subtraction").

The network's foreground is regenerated from /16 tokens by one linear
depatchify, so even where the subject is solid it is not the plate's own
pixels. A compositor's additive key does the opposite: estimate the backing
B behind the subject, then

    premultiplied FG = I - (1 - alpha) * B

which returns the plate exactly wherever alpha = 1 (grain, detail and bit
depth intact) and removes exactly the backing's contribution at edges. The
backing is estimated per frame from pixels the model calls background and
filled in under the subject with a pull-push (normalized pyramid) solve.

Where alpha is small, dividing by alpha amplifies any error in B, so the
model's own foreground takes over: the two are blended in premultiplied space
with weight smoothstep(lo, hi, alpha).
"""

import torch
import torch.nn.functional as F


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return -F.max_pool2d(-mask, 2 * radius + 1, stride=1, padding=radius)


def pull_push(x: torch.Tensor, w: torch.Tensor, min_size: int = 2) -> torch.Tensor:
    """Fill x where w == 0 from weighted neighbours at coarser scales.

    x: (N, C, H, W), w: (N, 1, H, W) in [0, 1]. Known pixels are returned
    unchanged; holes get a smooth membrane interpolation of the boundary.
    """
    levels = []
    xw, ww = x * w, w
    while min(ww.shape[-2:]) > min_size:
        levels.append((xw, ww))
        xw = F.avg_pool2d(xw, 2, ceil_mode=True)
        ww = F.avg_pool2d(ww, 2, ceil_mode=True)
    est = xw / ww.clamp(min=1e-8)
    # Coarsest level may still have empty cells: fall back to the global mean.
    glob = (x * w).sum((-2, -1), keepdim=True) / w.sum((-2, -1), keepdim=True).clamp(min=1e-8)
    est = torch.where(ww > 1e-8, est, glob.expand_as(est))
    for xw_l, ww_l in reversed(levels):
        up = F.interpolate(est, size=xw_l.shape[-2:], mode="bilinear", align_corners=False)
        c = ww_l.clamp(0.0, 1.0)
        est = c * (xw_l / ww_l.clamp(min=1e-8)) + (1.0 - c) * up
    return est


def estimate_backing(plate_linear: torch.Tensor, alpha: torch.Tensor,
                     bg_thresh: float = 0.004, erode_px: int = 2) -> torch.Tensor:
    known = _erode((alpha < bg_thresh).to(plate_linear.dtype), erode_px)
    return pull_push(plate_linear, known)


def _smoothstep(lo: float, hi: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def plate_foreground(plate_linear: torch.Tensor, alpha: torch.Tensor,
                     fg_model_linear: torch.Tensor, lo: float = 0.05, hi: float = 0.6,
                     backing: torch.Tensor | None = None) -> torch.Tensor:
    """Straight (un-premultiplied) linear foreground.

    plate_linear, fg_model_linear: (N, 3, H, W) scene-linear; alpha: (N, 1, H, W).
    """
    plate_linear = plate_linear.float()
    alpha = alpha.float().clamp(0.0, 1.0)
    fg_model_linear = fg_model_linear.float()
    if backing is None:
        backing = estimate_backing(plate_linear, alpha)
    premult_plate = plate_linear - (1.0 - alpha) * backing
    w = _smoothstep(lo, hi, alpha)
    premult = w * premult_plate + (1.0 - w) * alpha * fg_model_linear
    return torch.where(alpha > 1e-3, premult / alpha.clamp(min=1e-3), fg_model_linear)
