#!/usr/bin/env python3
"""Split each shot's composite error into an alpha part and a foreground part.

  alpha-only:  composite(pred alpha, GT foreground)  vs GT composite
  fg-only:     composite(GT alpha, pred foreground)  vs GT composite
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bench import lin_to_srgb, eval_backgrounds  # noqa: E402
from side_by_side_video import read_preds, read_plate  # noqa: E402


def comp_err(a, f, a_gt, f_gt, bgs):
    e = []
    for bg in bgs:
        gt = lin_to_srgb(a_gt[..., None] * f_gt + (1 - a_gt[..., None]) * bg).clip(0, 1)
        pr = lin_to_srgb(a[..., None] * f + (1 - a[..., None]) * bg).clip(0, 1)
        e.append(np.abs(pr - gt).mean() * 255)
    return float(np.mean(e))


def run(case_dir, res_dir, stride=8):
    gt_a = np.load(case_dir / "gt_alpha.npy", mmap_mode="r")
    gt_f = np.load(case_dir / "gt_fg_linear.npy", mmap_mode="r")
    A, Fs = read_preds(res_dir)
    bgs = eval_backgrounds(*gt_a.shape[1:])
    meta = json.loads((case_dir / "case.json").read_text())
    plates = read_plate(case_dir, meta)
    B = np.load(case_dir / "backing_linear.npy").astype(np.float32)
    r = {"both": [], "alpha_only": [], "fg_only": [], "fg_floor": []}
    for t in range(0, min(len(A), gt_a.shape[0]), stride):
        a_gt, f_gt = gt_a[t].astype(np.float32), gt_f[t].astype(np.float32)
        r["both"].append(comp_err(A[t], Fs[t], a_gt, f_gt, bgs))
        r["alpha_only"].append(comp_err(A[t], f_gt, a_gt, f_gt, bgs))
        r["fg_only"].append(comp_err(a_gt, Fs[t], a_gt, f_gt, bgs))
        P = plates[t]
        I = P if meta.get("linear_input") else np.where(P <= 0.04045, P / 12.92, ((np.clip(P, 0, None) + 0.055) / 1.055) ** 2.4)
        f_pl = np.where(a_gt[..., None] > 1e-3, (I - (1 - a_gt[..., None]) * B) / np.maximum(a_gt, 1e-3)[..., None], 0)
        r["fg_floor"].append(comp_err(a_gt, f_pl.astype(np.float32), a_gt, f_gt, bgs))
    return {k: float(np.mean(v)) for k, v in r.items()}


if __name__ == "__main__":
    bench, results = Path(sys.argv[1]), Path(sys.argv[2])
    cases = sorted(p.name for p in bench.iterdir() if (p / "case.json").exists() and (results / p.name).exists())
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(8) as ex:
        res = dict(zip(cases, ex.map(run, [bench / c for c in cases], [results / c for c in cases])))
    print(f"{'case':24s}{'both':>8s}{'alpha':>8s}{'fg':>8s}{'floor':>8s}")
    for c, v in res.items():
        print(f"{c:24s}{v['both']:8.3f}{v['alpha_only']:8.3f}{v['fg_only']:8.3f}{v['fg_floor']:8.3f}")
