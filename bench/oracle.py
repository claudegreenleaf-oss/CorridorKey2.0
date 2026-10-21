#!/usr/bin/env python3
"""Oracle checks for the enhanced design's two load-bearing assumptions.

1. Backing: plate-recovered foreground with the TRUE clean screen instead of
   the pull-push estimate. If this barely beats the estimate, a learned or
   accumulated backing memory is not worth building.
2. Physics alpha: per-pixel projection onto the line from the screen colour B
   to a smooth local foreground colour F_loc,

       alpha_phys = <I - B, F_loc - B> / (|F_loc - B|^2 + lam * sigma^2)

   blended with the network alpha by keying SNR. Two variants:
     oracle  B = true backing, F_loc = plate colour of GT-solid pixels, pushed out
     model   B = pull-push estimate from the model's alpha, F_loc from the model's
             solid pixels; both use the model's own outputs only.
   If the oracle variant does not beat the network alpha on hair, fur and the
   web, the /1 projection head is dropped from the design.

  python bench/oracle.py --bench B --results R/all --cases c14_animal_fur c15_hair_4k ...
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bench import lin_to_srgb, eval_backgrounds  # noqa: E402
from side_by_side_video import read_plate, read_preds  # noqa: E402


def srgb_to_lin(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def pull_push(x, w):
    """Normalized-pyramid fill of x where w == 0 (numpy, HxWxC)."""
    levels = []
    xw, ww = x * w[..., None], w.astype(np.float32)
    while min(ww.shape) > 4:
        levels.append((xw, ww))
        xw = cv2.resize(xw, (max(1, xw.shape[1] // 2), max(1, xw.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        ww = cv2.resize(ww, (max(1, ww.shape[1] // 2), max(1, ww.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    est = xw / np.maximum(ww, 1e-8)[..., None]
    for xw_l, ww_l in reversed(levels):
        up = cv2.resize(est, (xw_l.shape[1], xw_l.shape[0]), interpolation=cv2.INTER_LINEAR)
        c = np.clip(ww_l, 0, 1)[..., None]
        est = c * (xw_l / np.maximum(ww_l, 1e-8)[..., None]) + (1 - c) * up
    return est


def erode(mask, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask.astype(np.uint8), k) > 0


def physics_alpha(I, B, F_loc, sigma, lam=1.0):
    d = F_loc - B
    num = ((I - B) * d).sum(-1)
    den = (d * d).sum(-1) + lam * sigma ** 2
    a = np.clip(num / np.maximum(den, 1e-8), 0, 1)
    snr = (d * d).sum(-1) / max(sigma ** 2, 1e-10)
    return a, snr


def fuse(a_phys, a_net, snr, k=1.0):
    """SNR fusion weight w = s / (s + k^2 * c): trust physics where alpha noise
    sigma/|F-B| is small. c = 400 puts w = 0.5 at a projection noise of 0.05."""
    w = snr / (snr + 400.0 * k)
    return w * a_phys + (1 - w) * a_net, w


def metrics(a, a_gt, f, f_gt, bgs, band):
    out = {"sad": float(np.abs(a - a_gt).sum() / 1000),
           "edge_mae": float(np.abs(a - a_gt)[band].mean()) if band.any() else 0.0}
    e = []
    for bg in bgs:
        gt = lin_to_srgb(a_gt[..., None] * f_gt + (1 - a_gt[..., None]) * bg).clip(0, 1)
        pr = lin_to_srgb(a[..., None] * f + (1 - a[..., None]) * bg).clip(0, 1)
        e.append(np.abs(pr - gt).mean() * 255)
    out["comp_mae"] = float(np.mean(e))
    return out


def run_case(case_dir: Path, res_dir: Path, stride: int):
    meta = json.loads((case_dir / "case.json").read_text())
    gt_a = np.load(case_dir / "gt_alpha.npy", mmap_mode="r")
    gt_f = np.load(case_dir / "gt_fg_linear.npy", mmap_mode="r")
    B_true = np.load(case_dir / "backing_linear.npy").astype(np.float32)
    plates = read_plate(case_dir, meta)
    a_net_all, f_net_all = read_preds(res_dir)
    H, W = gt_a.shape[1:]
    bgs = eval_backgrounds(H, W)
    rows = {k: [] for k in ("net", "plate_fg_estB", "plate_fg_trueB",
                            "phys_oracle", "fused_oracle", "phys_model", "fused_model")}
    wstats = []
    T = min(len(plates), len(a_net_all), gt_a.shape[0])
    for t in range(0, T, stride):
        a_gt = gt_a[t].astype(np.float32)
        f_gt = gt_f[t].astype(np.float32)
        P = plates[t]
        I = P if meta.get("linear_input") else srgb_to_lin(P)
        a_net, f_net = a_net_all[t], f_net_all[t]
        trans = (a_gt > 1e-3) & (a_gt < 1 - 1e-3)
        band = cv2.dilate(trans.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

        rows["net"].append(metrics(a_net, a_gt, f_net, f_gt, bgs, band))

        # noise sigma from the screen: residual against a smoothed screen
        bg_mask = erode(a_gt < 1e-3, 6)
        resid = (I - cv2.GaussianBlur(I, (0, 0), 3))[bg_mask]
        sigma = float(np.median(np.abs(resid)) * 1.4826 * 1.2) if resid.size else 0.01

        # --- oracle 1: plate foreground with true vs estimated backing
        B_est = pull_push(I, erode(a_net < 0.004, 2).astype(np.float32))
        for name, B in (("plate_fg_estB", B_est), ("plate_fg_trueB", B_true)):
            prem = I - (1 - a_net[..., None]) * B
            f = np.where(a_net[..., None] > 1e-3, prem / np.maximum(a_net, 1e-3)[..., None], f_net)
            rows[name].append(metrics(a_net, a_gt, f, f_gt, bgs, band))

        # --- oracle 2: physics alpha
        solid_gt = erode(a_gt > 0.995, 2).astype(np.float32)
        F_loc_o = pull_push(I, solid_gt)
        a_ph, snr = physics_alpha(I, B_true, F_loc_o, sigma)
        a_fu, w = fuse(a_ph, a_net, snr)
        wstats.append(float(w[band].mean()) if band.any() else 0.0)
        for name, a in (("phys_oracle", a_ph), ("fused_oracle", a_fu)):
            prem = I - (1 - a[..., None]) * B_true
            f = np.where(a[..., None] > 1e-3, prem / np.maximum(a, 1e-3)[..., None], f_net)
            rows[name].append(metrics(a, a_gt, f, f_gt, bgs, band))

        solid_m = erode(a_net > 0.995, 2).astype(np.float32)
        F_loc_m = pull_push(I, solid_m)
        a_pm, snr_m = physics_alpha(I, B_est, F_loc_m, sigma)
        a_fm, _ = fuse(a_pm, a_net, snr_m)
        for name, a in (("phys_model", a_pm), ("fused_model", a_fm)):
            prem = I - (1 - a[..., None]) * B_est
            f = np.where(a[..., None] > 1e-3, prem / np.maximum(a, 1e-3)[..., None], f_net)
            rows[name].append(metrics(a, a_gt, f, f_gt, bgs, band))

    agg = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in rows.items()}
    agg["_mean_fusion_weight_in_band"] = float(np.mean(wstats))
    agg["_sigma"] = sigma
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=len(args.cases)) as ex:
        futs = {c: ex.submit(run_case, args.bench / c, args.results / c, args.stride) for c in args.cases}
        res = {c: f.result() for c, f in futs.items()}
    args.out.write_text(json.dumps(res, indent=1))
    keys = ["net", "plate_fg_estB", "plate_fg_trueB", "phys_oracle", "fused_oracle", "phys_model", "fused_model"]
    for m in ("comp_mae", "edge_mae", "sad"):
        print(f"\n## {m}")
        print(f"{'case':24s}" + "".join(f"{k:>16s}" for k in keys) + f"{'w_band':>8s}")
        for c, r in res.items():
            print(f"{c:24s}" + "".join(f"{r[k][m]:16.4f}" for k in keys) + f"{r['_mean_fusion_weight_in_band']:8.2f}")


if __name__ == "__main__":
    main()
