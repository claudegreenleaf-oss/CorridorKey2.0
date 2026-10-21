#!/usr/bin/env python3
"""Visual side-by-side: released CorridorKey vs a variant, against exact GT.

For each case, one image:
  row 1  plate | released composite | variant composite | ground truth
  row 2  4x zoom of the window where the released model's error is largest
  row 3  composite error maps (x8), released vs variant, same scale

Composites are over 18% grey, where fringing and holes are easiest to see.

  python bench/side_by_side.py --bench B --results R --a baseline --b all --out sbs/
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from eval_bench import find_outputs, lin_to_srgb

GREY = 0.18
TILE_W = 640


def read_plate_frame(case_dir: Path, meta: dict, t: int):
    p = case_dir / meta["plate"]
    if p.is_dir():
        f = sorted(p.iterdir())[t]
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32)
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        if f.suffix == ".exr":
            return np.clip(lin_to_srgb(img), 0, 1)
        return img / (65535.0 if img.max() > 255 else 255.0)
    W, H = meta["width"], meta["height"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf", f"select=eq(n\\,{t})",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<u2").reshape(H, W, 3).astype(np.float32) / 65535.0


def read_pred(out_case: Path, t: int):
    a_dir, f_dir = find_outputs(out_case)
    a = cv2.imread(str(sorted(a_dir.glob("*.exr"))[t]), cv2.IMREAD_UNCHANGED).astype(np.float32)
    a = a if a.ndim == 2 else a[..., 0]
    f = cv2.imread(str(sorted(f_dir.glob("*.exr"))[t]), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return np.clip(a, 0, 1), cv2.cvtColor(f[..., :3], cv2.COLOR_BGR2RGB)


def comp(a, f):
    return np.clip(lin_to_srgb(a[..., None] * f + (1 - a[..., None]) * GREY), 0, 1)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def to8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def heat(err, scale=8.0):
    e = np.clip(err.mean(-1) * scale, 0, 1)
    return cv2.cvtColor(cv2.applyColorMap(to8(e), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def worst_window(err, win):
    s = cv2.boxFilter(err.mean(-1), -1, (win, win), normalize=True)
    y, x = np.unravel_index(np.argmax(s), s.shape)
    H, W = s.shape
    y0 = int(np.clip(y - win // 2, 0, H - win))
    x0 = int(np.clip(x - win // 2, 0, W - win))
    return y0, x0


def render_case(case_dir, ra, rb, names, out_dir, t=None):
    meta = json.loads((case_dir / "case.json").read_text())
    gt_a = np.load(case_dir / "gt_alpha.npy", mmap_mode="r")
    gt_f = np.load(case_dir / "gt_fg_linear.npy", mmap_mode="r")
    t = gt_a.shape[0] // 2 if t is None else t
    a_gt, f_gt = gt_a[t].astype(np.float32), gt_f[t].astype(np.float32)
    plate = read_plate_frame(case_dir, meta, t)
    a1, f1 = read_pred(ra, t)
    a2, f2 = read_pred(rb, t)
    c_gt, c1, c2 = comp(a_gt, f_gt), comp(a1, f1), comp(a2, f2)
    e1, e2 = np.abs(c1 - c_gt), np.abs(c2 - c_gt)

    H, W = a_gt.shape
    s = TILE_W / W
    th = int(H * s)
    small = lambda x: cv2.resize(to8(x), (TILE_W, th), interpolation=cv2.INTER_AREA)
    m1, m2 = e1.mean() * 255, e2.mean() * 255
    row1 = np.hstack([label(small(plate), "plate"),
                      label(small(c1), f"{names[0]}  err {m1:.2f}"),
                      label(small(c2), f"{names[1]}  err {m2:.2f}"),
                      label(small(c_gt), "ground truth")])

    win = max(64, min(H, W) // 6)
    y0, x0 = worst_window(e1, win)
    crop = lambda x: cv2.resize(to8(x[y0:y0 + win, x0:x0 + win]), (th, th),
                                interpolation=cv2.INTER_NEAREST)
    pad = lambda img: np.pad(img, ((0, 0), (0, TILE_W - img.shape[1]), (0, 0)))
    row2 = np.hstack([label(pad(crop(plate)), f"zoom x{th / win:.1f}"),
                      label(pad(crop(c1)), names[0]),
                      label(pad(crop(c2)), names[1]),
                      label(pad(crop(c_gt)), "ground truth")])
    he1, he2 = heat(e1), heat(e2)
    row3 = np.hstack([label(cv2.resize(he1, (TILE_W, th)), f"{names[0]} error x8"),
                      label(cv2.resize(he2, (TILE_W, th)), f"{names[1]} error x8"),
                      label(pad(cv2.resize(he1[y0:y0 + win, x0:x0 + win], (th, th),
                                           interpolation=cv2.INTER_NEAREST)), f"{names[0]} error, zoom"),
                      label(pad(cv2.resize(he2[y0:y0 + win, x0:x0 + win], (th, th),
                                           interpolation=cv2.INTER_NEAREST)), f"{names[1]} error, zoom")])
    sheet = np.vstack([row1, row2, row3])
    out = out_dir / f"{case_dir.name}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"case": case_dir.name, "frame": t, "err_a": float(m1), "err_b": float(m2),
            "why": meta.get("why", ""), "image": out.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b", default="all")
    ap.add_argument("--names", nargs=2, default=["released", "ours"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_dir in sorted(p for p in args.bench.iterdir() if (p / "case.json").exists()):
        ra, rb = args.results / args.a / case_dir.name, args.results / args.b / case_dir.name
        if not (ra.exists() and rb.exists()):
            continue
        rows.append(render_case(case_dir, ra, rb, args.names, args.out))
        print(rows[-1], flush=True)
    (args.out / "index.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
