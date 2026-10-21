#!/usr/bin/env python3
"""Native-resolution, frame-locked videos for the interactive comparison viewer.

Per case it writes, at the plate's native resolution:
  plate.mp4                      the plate as decoded
  <method>_comp.mp4, gt_comp.mp4 composite over 18% grey
  <method>_alpha.mp4, gt_alpha.mp4  the matte
and a manifest.json listing cases, methods, sizes and scores.

Encoding: H.264 High, CRF 14, keyframe every 4 frames so stepping and
seeking land exactly; yuv420p for universal browser playback.

  python bench/viewer_videos.py --bench B --results R --methods baseline:Released all:Ours --out viewer/
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval_bench import lin_to_srgb
from side_by_side_video import read_plate, read_preds

GREY = 0.18


def enc(path, frames, fps, gray=False):
    H, W = frames[0].shape[:2]
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "14", "-g", "4", "-keyint_min", "4", "-sc_threshold", "0", "-bf", "0",
         "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-colorspace", "bt709", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(path)


def to8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def comp(a, f):
    return to8(lin_to_srgb(a[..., None] * f + (1 - a[..., None]) * GREY))


def do_case(case_dir, results, methods, out):
    meta = json.loads((case_dir / "case.json").read_text())
    od = out / case_dir.name
    od.mkdir(parents=True, exist_ok=True)
    fps = float(meta.get("fps", 24))
    gt_a = np.load(case_dir / "gt_alpha.npy", mmap_mode="r")
    gt_f = np.load(case_dir / "gt_fg_linear.npy", mmap_mode="r")
    plates = read_plate(case_dir, meta)
    T = min(len(plates), gt_a.shape[0])
    enc(od / "plate.mp4", [to8(p) for p in plates[:T]], fps)
    enc(od / "gt_comp.mp4", [comp(gt_a[t].astype(np.float32), gt_f[t].astype(np.float32)) for t in range(T)], fps)
    enc(od / "gt_alpha.mp4", [to8(gt_a[t].astype(np.float32)) for t in range(T)], fps, gray=True)
    scores = {}
    for key, _label in methods:
        A, Fs = read_preds(results / key / case_dir.name)
        enc(od / f"{key}_comp.mp4", [comp(A[t], Fs[t]) for t in range(T)], fps)
        enc(od / f"{key}_alpha.mp4", [to8(A[t]) for t in range(T)], fps, gray=True)
        s = json.loads((results / key / "scores.json").read_text()).get(case_dir.name, {})
        scores[key] = {m: round(s[m], 4) for m in ("comp_mae", "comp_psnr", "edge_mae", "holes", "dtssd", "fg_mae") if m in s}
    return {"case": case_dir.name, "why": meta.get("why", ""), "width": meta["width"],
            "height": meta["height"], "frames": T, "fps": fps, "scores": scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="key:Label, results/<key> must exist")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    methods = [tuple(m.split(":", 1)) for m in args.methods]
    cases = sorted(p for p in args.bench.iterdir() if (p / "case.json").exists())
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(do_case, cases, [args.results] * len(cases), [methods] * len(cases),
                           [args.out] * len(cases)))
    (args.out / "manifest.json").write_text(json.dumps(
        {"methods": [{"key": k, "label": l} for k, l in methods], "cases": rows}, indent=1))
    print("wrote", len(rows), "cases")


if __name__ == "__main__":
    main()
