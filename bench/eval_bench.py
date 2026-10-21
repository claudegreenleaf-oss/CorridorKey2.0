#!/usr/bin/env python3
"""Score CorridorKey outputs against the hard benchmark's exact ground truth.

Reads, per case, the EXR sequences written by
`infer.py --output_format exr --outputs alpha fg` and reports:

  comp_mae      composite error over four new backgrounds, 8-bit sRGB units,
                whole frame (the headline: what a compositor actually sees)
  comp_edge     the same, restricted to the edge band
  comp_psnr     composite PSNR (dB), sRGB display, averaged over backgrounds
  sad / mse     standard alpha metrics (SAD / 1000, MSE x 1000)
  grad          gradient error (Gaussian derivative, sigma 1.4), / 1000
  edge_mae      mean |alpha error| inside the band where 0 < gt < 1, dilated 7 px
  holes         mean (1 - alpha) where gt is solid (eroded 7 px), x 1000
  leak          mean alpha where gt is empty (eroded 7 px), x 1000
  dtssd         temporal coherence error (Erofeev et al.), x 100
  fg_mae        foreground color error, sRGB 8-bit units, alpha-weighted where gt > 0.05

Usage:
  python bench/eval_bench.py --bench ~/corridorkey/bench --results ~/corridorkey/results/baseline
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

METRICS = ["comp_mae", "comp_edge", "comp_psnr", "sad", "mse", "grad", "edge_mae",
           "holes", "leak", "dtssd", "fg_mae"]
LOWER_IS_BETTER = {m: m != "comp_psnr" for m in METRICS}


def lin_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def eval_backgrounds(H, W):
    """Four fixed new backgrounds, scene-linear."""
    rng = np.random.default_rng(1234)
    grey = np.full((H, W, 3), 0.18, np.float32)
    dark = np.full((H, W, 3), 0.01, np.float32)
    bright = np.full((H, W, 3), 0.75, np.float32)
    # colourful, high-contrast texture: smooth color fields plus fine detail
    g = rng.random((H // 32 + 2, W // 32 + 2, 3)).astype(np.float32)
    tex = cv2.resize(g, (W + 64, H + 64), interpolation=cv2.INTER_CUBIC)[32:32 + H, 32:32 + W]
    fine = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 1.0)[..., None]
    tex = np.clip(tex * 0.8 + (fine - 0.5) * 0.6, 0, 1) ** 2.2
    return [grey, dark, bright, tex.astype(np.float32)]


def read_exr_seq(d: Path, channels: int):
    files = sorted(d.glob("*.exr"))
    out = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cannot read {f}")
        img = img.astype(np.float32)
        if channels == 1:
            out.append(img if img.ndim == 2 else img[..., 0])
        else:
            out.append(cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB))
    return out


def find_outputs(case_dir: Path):
    a = sorted(case_dir.glob("*_alpha_exr"))
    f = sorted(case_dir.glob("*_fg_exr"))
    if not a or not f:
        raise FileNotFoundError(f"no alpha/fg EXR output in {case_dir}")
    return a[0], f[0]


def gauss_grad(a):
    gx = ndimage.gaussian_filter(a, 1.4, order=(0, 1))
    gy = ndimage.gaussian_filter(a, 1.4, order=(1, 0))
    return np.sqrt(gx ** 2 + gy ** 2)


def score_case(bench_case: Path, out_case: Path, frame_stride: int = 1):
    meta = json.loads((bench_case / "case.json").read_text())
    gt_a = np.load(bench_case / "gt_alpha.npy", mmap_mode="r")
    gt_f = np.load(bench_case / "gt_fg_linear.npy", mmap_mode="r")
    a_dir, f_dir = find_outputs(out_case)
    pa_all = read_exr_seq(a_dir, 1)
    pf_all = read_exr_seq(f_dir, 3)
    T = min(len(pa_all), gt_a.shape[0])
    H, W = gt_a.shape[1:]
    bgs = eval_backgrounds(H, W)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    acc = {m: [] for m in METRICS}
    prev = None
    for t in range(0, T, frame_stride):
        a = gt_a[t].astype(np.float32)
        f = gt_f[t].astype(np.float32)
        pa = np.clip(pa_all[t], 0, 1)
        pf = pf_all[t]
        if pa.shape != a.shape:
            raise ValueError(f"{bench_case.name}: shape {pa.shape} != gt {a.shape}")

        trans = ((a > 1e-3) & (a < 1 - 1e-3)).astype(np.uint8)
        band = cv2.dilate(trans, k) > 0
        solid = cv2.erode((a >= 1 - 1e-3).astype(np.uint8), k) > 0
        empty = cv2.erode((a <= 1e-3).astype(np.uint8), k) > 0
        err = pa - a

        acc["sad"].append(np.abs(err).sum() / 1000)
        acc["mse"].append((err ** 2).mean() * 1000)
        acc["grad"].append(((gauss_grad(pa) - gauss_grad(a)) ** 2).sum() / 1000)
        acc["edge_mae"].append(np.abs(err[band]).mean() if band.any() else 0.0)
        acc["holes"].append((1 - pa[solid]).mean() * 1000 if solid.any() else 0.0)
        acc["leak"].append(pa[empty].mean() * 1000 if empty.any() else 0.0)

        if prev is not None and frame_stride == 1:
            d = (pa - prev[0]) - (a - prev[1])
            acc["dtssd"].append(np.sqrt((d ** 2).mean()) * 100)
        prev = (pa, a)

        maes, edges, psnrs = [], [], []
        for bg in bgs:
            gt_c = lin_to_srgb(a[..., None] * f + (1 - a[..., None]) * bg).clip(0, 1)
            pr_c = lin_to_srgb(pa[..., None] * pf + (1 - pa[..., None]) * bg).clip(0, 1)
            e = np.abs(pr_c - gt_c)
            maes.append(e.mean() * 255)
            edges.append(e[band].mean() * 255 if band.any() else 0.0)
            mse = ((pr_c - gt_c) ** 2).mean()
            psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
        acc["comp_mae"].append(np.mean(maes))
        acc["comp_edge"].append(np.mean(edges))
        acc["comp_psnr"].append(np.mean(psnrs))

        w = a * (a > 0.05)
        if w.sum() > 0:
            fe = np.abs(lin_to_srgb(pf).clip(0, 1) - lin_to_srgb(f).clip(0, 1)).mean(-1)
            acc["fg_mae"].append((fe * w).sum() / w.sum() * 255)

    return {m: float(np.mean(v)) if v else float("nan") for m, v in acc.items()}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--frame_stride", type=int, default=1)
    args = ap.parse_args()

    scores = {}
    for case_dir in sorted(p for p in args.bench.iterdir() if (p / "case.json").exists()):
        if args.only and case_dir.name not in args.only:
            continue
        out_case = args.results / case_dir.name
        if not out_case.exists():
            print(f"skip {case_dir.name}: no results")
            continue
        s, _ = score_case(case_dir, out_case, args.frame_stride)
        scores[case_dir.name] = s
        print(f"{case_dir.name:26s} " + " ".join(f"{m}={s[m]:.3f}" for m in METRICS), flush=True)
    (args.results / "scores.json").write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
