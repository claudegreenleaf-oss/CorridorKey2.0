#!/usr/bin/env python3
"""Side-by-side comparison videos: released CorridorKey vs a variant, vs exact GT.

Per case, two looping H.264 clips (browser-playable):
  <case>_full.mp4  2x2 grid: plate | released / ours | ground truth
  <case>_zoom.mp4  the same grid on a fixed window around the region that is
                   hardest for both models (edge band, whole clip), nearest-
                   neighbour upscaled and played at half speed, so edge
                   flicker and temporal coherence are easy to judge.
Composites are over 18% grey.

  python bench/side_by_side_video.py --bench B --results R --a baseline --b all --out sbsv/
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
PANEL = (960, 540)


def read_plate(case_dir: Path, meta: dict):
    p = case_dir / meta["plate"]
    H, W = meta["height"], meta["width"]
    if p.is_dir():
        out = []
        for f in sorted(p.iterdir()):
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32)
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
            out.append(np.clip(lin_to_srgb(img), 0, 1) if f.suffix == ".exr"
                       else img / (65535.0 if img.max() > 255 else 255.0))
        return out
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "rawvideo",
                          "-pix_fmt", "rgb48le", "-"], capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, "<u2").reshape(-1, H, W, 3)
    return [a.astype(np.float32) / 65535.0 for a in arr]


def read_preds(out_case: Path):
    a_dir, f_dir = find_outputs(out_case)
    A, Fs = [], []
    for fa, ff in zip(sorted(a_dir.glob("*.exr")), sorted(f_dir.glob("*.exr"))):
        a = cv2.imread(str(fa), cv2.IMREAD_UNCHANGED).astype(np.float32)
        A.append(np.clip(a if a.ndim == 2 else a[..., 0], 0, 1))
        f = cv2.imread(str(ff), cv2.IMREAD_UNCHANGED).astype(np.float32)
        Fs.append(cv2.cvtColor(f[..., :3], cv2.COLOR_BGR2RGB))
    return A, Fs


def comp(a, f):
    return np.clip(lin_to_srgb(a[..., None] * f + (1 - a[..., None]) * GREY), 0, 1)


def to8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grid(panels):
    return np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])


class Writer:
    def __init__(self, path, fps, size):
        W, H = size
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, rgb8):
        self.p.stdin.write(np.ascontiguousarray(rgb8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def render_case(case_dir, ra, rb, names, out_dir):
    meta = json.loads((case_dir / "case.json").read_text())
    gt_a = np.load(case_dir / "gt_alpha.npy", mmap_mode="r")
    gt_f = np.load(case_dir / "gt_fg_linear.npy", mmap_mode="r")
    plates = read_plate(case_dir, meta)
    a1, f1 = read_preds(ra)
    a2, f2 = read_preds(rb)
    T = min(len(plates), len(a1), len(a2), gt_a.shape[0])
    H, W = gt_a.shape[1:]

    # Fixed zoom window: most combined composite error inside the GT edge
    # band, accumulated over the whole clip (fair to both models).
    wh = max(54, H // 6)
    ww = int(wh * PANEL[0] / PANEL[1])
    acc = np.zeros((H, W), np.float32)
    comps = []
    for t in range(T):
        a, f = gt_a[t].astype(np.float32), gt_f[t].astype(np.float32)
        cg, c1, c2 = comp(a, f), comp(a1[t], f1[t]), comp(a2[t], f2[t])
        comps.append((cg, c1, c2))
        band = cv2.dilate(((a > 1e-3) & (a < 1 - 1e-3)).astype(np.uint8), np.ones((9, 9), np.uint8))
        acc += (np.abs(c1 - cg) + np.abs(c2 - cg)).mean(-1) * band
    score = cv2.boxFilter(acc, -1, (ww, wh), normalize=True)
    y, x = np.unravel_index(np.argmax(score), score.shape)
    y0 = int(np.clip(y - wh // 2, 0, H - wh))
    x0 = int(np.clip(x - ww // 2, 0, W - ww))

    fps = float(meta.get("fps", 24))
    full = Writer(out_dir / f"{case_dir.name}_full.mp4", fps, (2 * PANEL[0], 2 * PANEL[1]))
    zoom = Writer(out_dir / f"{case_dir.name}_zoom.mp4", fps / 2, (2 * PANEL[0], 2 * PANEL[1]))
    small = lambda x: cv2.resize(to8(x), PANEL, interpolation=cv2.INTER_AREA)
    crop = lambda x: cv2.resize(to8(x[y0:y0 + wh, x0:x0 + ww]), PANEL, interpolation=cv2.INTER_NEAREST)
    zf = PANEL[1] / wh
    for t in range(T):
        cg, c1, c2 = comps[t]
        e1 = np.abs(c1 - cg).mean() * 255
        e2 = np.abs(c2 - cg).mean() * 255
        tags = ["plate", f"{names[0]}  err {e1:.2f}", f"{names[1]}  err {e2:.2f}", "ground truth"]
        full.write(grid([label(small(x), f"{tag}   frame {t}") for x, tag in zip((plates[t], c1, c2, cg), tags)]))
        zoom.write(grid([label(crop(x), f"{tag}   zoom x{zf:.1f}") for x, tag in
                         zip((plates[t], c1, c2, cg), ["plate", names[0], names[1], "ground truth"])]))
    full.close()
    zoom.close()
    return {"case": case_dir.name, "frames": T, "why": meta.get("why", ""),
            "full": f"{case_dir.name}_full.mp4", "zoom": f"{case_dir.name}_zoom.mp4",
            "zoom_window": [int(x0), int(y0), int(ww), int(wh)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b", default="all")
    ap.add_argument("--names", nargs=2, default=["released", "ours"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for case_dir in sorted(p for p in args.bench.iterdir() if (p / "case.json").exists()):
        if args.only and case_dir.name not in args.only:
            continue
        ra, rb = args.results / args.a / case_dir.name, args.results / args.b / case_dir.name
        if ra.exists() and rb.exists():
            jobs.append((case_dir, ra, rb))
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(render_case, *zip(*jobs), [args.names] * len(jobs), [args.out] * len(jobs)))
    (args.out / "index.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r["case"], r["zoom_window"])


if __name__ == "__main__":
    main()
