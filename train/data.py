"""On-the-fly keying clips with exact ground truth, for fine-tuning.

Each sample is a short clip (T frames) cut from a real foreground with its
alpha, placed on a randomized keying stage, run through a randomized camera,
and pushed through a real ffmpeg encode/decode so codec damage is genuine.
It uses the same stage and camera simulator as the benchmark
(bench/make_bench.py), but only the *train* splits:

  VideoMatte240K_JPEG_HD/train   (the benchmark uses /test)
  Transparent-460/Train          (the benchmark uses /Test)

Targets: alpha and the noise-free, spill-free scene-linear foreground.
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from make_bench import (  # noqa: E402
    SCREEN, add_spill, affine, camera_noise, disk_kernel, green_wardrobe,
    lin_to_srgb, make_backing, sharpen, smoke_layer, srgb_to_lin, warp,
    _read_alpha, _read_rgb,
)

cv2.setNumThreads(1)


class _VMIndex:
    def __init__(self, root: Path):
        self.clips = []
        for c in sorted((root / "fgr").iterdir()):
            fgr = sorted(c.iterdir())
            pha = sorted((root / "pha" / c.name).iterdir())
            if len(fgr) >= 8 and len(fgr) == len(pha):
                self.clips.append((fgr, pha))


def _encode_decode(frames_display, kind, crf, rng):
    """Round-trip display-coded frames through a real codec; returns float32 frames."""
    T = len(frames_display)
    H, W = frames_display[0].shape[:2]
    if kind == "png16":
        return [np.round(np.clip(f, 0, 1) * 65535) / 65535 for f in frames_display]
    if kind == "prores10":
        enc = ["-c:v", "prores_ks", "-profile:v", str(rng.choice([2, 3])),
               "-pix_fmt", "yuv422p10le"]
        ext = ".mov"
    else:
        enc = ["-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
               "-pix_fmt", "yuv420p"]
        ext = ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, dir="/dev/shm" if os.path.isdir("/dev/shm") else None,
                                      delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le",
           "-s", f"{W}x{H}", "-r", "24", "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int",
           *enc, "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-y", tmp.name]
    raw = b"".join((np.clip(f, 0, 1) * 65535 + 0.5).astype("<u2").tobytes() for f in frames_display)
    try:
        subprocess.run(cmd, input=raw, capture_output=True, check=True)
        out = _decode(tmp.name)
    finally:
        os.unlink(tmp.name)
    arr = np.frombuffer(out, dtype="<u2")[:T * H * W * 3].reshape(T, H, W, 3)
    return list(arr.astype(np.float32) / 65535.0)


def _decode(path):
    dec = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", "scale=in_color_matrix=bt709:in_range=tv:"
                  "flags=spline+accurate_rnd+full_chroma_int+full_chroma_inp",
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
    return subprocess.run(dec, capture_output=True, check=True).stdout


class KeyingClips(Dataset):
    def __init__(self, data_root: Path, crop_hw=(640, 1152), T=4, length=10 ** 7,
                 t460_prob=0.12, seed=0):
        self.data_root = Path(data_root)
        self.crop_hw = crop_hw
        self.T = T
        self.length = length
        self.t460_prob = t460_prob
        self.seed = seed
        self.vm = _VMIndex(self.data_root / "VideoMatte240K_JPEG_HD" / "train")
        t460 = self.data_root / "t460" / "Train"
        self.t460 = sorted(p.stem for p in (t460 / "fg").glob("*.jpg")) if t460.exists() else []

    def __len__(self):
        return self.length

    # -- foreground clip -----------------------------------------------------
    def _vm_clip(self, rng):
        H, W = self.crop_hw
        fgr, pha = self.vm.clips[rng.integers(len(self.vm.clips))]
        step = int(rng.choice([1, 1, 2]))
        t0 = int(rng.integers(0, max(1, len(fgr) - step * self.T)))
        idx = [min(t0 + i * step, len(fgr) - 1) for i in range(self.T)]
        a0 = _read_alpha(pha[idx[0]])
        h, w = a0.shape
        # Detail scale: map a crop of the 4K source onto the training crop.
        s = float(rng.uniform(0.22, 0.6))
        ch, cw = min(h, int(H / s)), min(w, int(W / s))
        ys, xs = np.nonzero(a0 > 0.5)
        if ys.size:
            cy = int(np.clip(rng.choice(ys) + rng.normal(0, ch * 0.15), ch // 2, h - ch // 2))
            cx = int(np.clip(rng.choice(xs) + rng.normal(0, cw * 0.15), cw // 2, w - cw // 2))
        else:
            cy, cx = h // 2, w // 2
        y0, x0 = cy - ch // 2, cx - cw // 2
        flip = rng.random() < 0.5
        Ps, As = [], []
        for i in idx:
            fg = srgb_to_lin(_read_rgb(fgr[i])[y0:y0 + ch, x0:x0 + cw])
            a = (a0 if i == idx[0] else _read_alpha(pha[i]))[y0:y0 + ch, x0:x0 + cw]
            P = cv2.resize(fg * a[..., None], (W, H), interpolation=cv2.INTER_AREA)
            A = np.clip(cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA), 0, 1)
            if flip:
                P, A = P[:, ::-1].copy(), A[:, ::-1].copy()
            Ps.append(P)
            As.append(A)
        return Ps, As

    def _still_clip(self, rng):
        H, W = self.crop_hw
        name = self.t460[rng.integers(len(self.t460))]
        root = self.data_root / "t460" / "Train"
        fg = srgb_to_lin(_read_rgb(root / "fg" / f"{name}.jpg"))
        a = _read_alpha(root / "alpha" / f"{name}.png")
        h, w = a.shape
        s = float(rng.uniform(0.6, 1.1)) * min(H / h, W / w)
        P0 = cv2.resize(fg * a[..., None], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        A0 = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        canvasP = np.zeros((H, W, 3), np.float32)
        canvasA = np.zeros((H, W), np.float32)
        ph, pw = A0.shape
        oy, ox = (H - ph) // 2, (W - pw) // 2
        sy0, sx0 = max(0, -oy), max(0, -ox)
        dy0, dx0 = max(0, oy), max(0, ox)
        hh, ww = min(ph - sy0, H - dy0), min(pw - sx0, W - dx0)
        canvasP[dy0:dy0 + hh, dx0:dx0 + ww] = P0[sy0:sy0 + hh, sx0:sx0 + ww]
        canvasA[dy0:dy0 + hh, dx0:dx0 + ww] = A0[sy0:sy0 + hh, sx0:sx0 + ww]
        v = rng.normal(0, W * 0.004, 2)
        Ps, As = [], []
        for t in range(self.T):
            P, A = warp(canvasP, canvasA, affine(tx=v[0] * t, ty=v[1] * t, rot_deg=0.1 * t,
                                                 cx=W / 2, cy=H / 2))
            Ps.append(P)
            As.append(A)
        return Ps, As

    # -- stage, camera, codec ------------------------------------------------
    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, os.getpid()))
        H, W = self.crop_hw
        T = self.T
        if self.t460 and rng.random() < self.t460_prob:
            Ps, As = self._still_clip(rng)
        else:
            Ps, As = self._vm_clip(rng)

        screen = "blue" if rng.random() < 0.2 else "green"
        B = make_backing(H, W, rng, screen,
                         uneven=float(rng.uniform(0, 0.55)),
                         wrinkles=float(rng.uniform(0, 0.07)) if rng.random() < 0.6 else 0.0,
                         markers=int(rng.integers(0, 10)) if rng.random() < 0.5 else 0,
                         vignette=float(rng.uniform(0, 0.4)))
        B *= float(np.exp(rng.normal(0, 0.25)))           # screen brightness / hue drift
        B *= (1 + rng.normal(0, 0.08, 3)).astype(np.float32)
        exposure = 2.0 ** float(rng.uniform(-2.8, 0.8) if rng.random() < 0.5 else rng.normal(0, 0.3))
        spill = float(rng.uniform(0, 0.3))
        iso = float(rng.choice([0.0, rng.uniform(0.3, 2.0), rng.uniform(2.0, 7.0)], p=[0.15, 0.55, 0.3]))
        motion = rng.random() < 0.25
        defocus = float(rng.uniform(2, 12)) if rng.random() < 0.15 else 0.0
        smoke = float(rng.uniform(0.3, 0.8)) if rng.random() < 0.12 else 0.0
        wardrobe = float(rng.uniform(0.5, 0.9)) if rng.random() < 0.1 else 0.0
        kind = str(rng.choice(["png16", "prores10", "h264_8"], p=[0.15, 0.35, 0.5]))
        crf = int(rng.integers(14, 33))
        sharp = float(rng.uniform(0.3, 1.0)) if rng.random() < 0.25 else 0.0
        vel = rng.normal(0, W * 0.025, 2)
        smoke_rng = np.random.default_rng(int(rng.integers(1 << 31)))
        if hasattr(smoke_layer, "_cache"):
            del smoke_layer._cache

        disp, gt_a, gt_f = [], [], []
        for t in range(T):
            P, A = Ps[t], As[t]
            if wardrobe:
                P = green_wardrobe(P, A, wardrobe)
            if motion:
                subs = 8
                Pm, Am = np.zeros_like(P), np.zeros_like(A)
                for s in range(subs):
                    f = 0.5 * (s + 0.5) / subs
                    Pq, Aq = warp(P, A, affine(tx=vel[0] * f, ty=vel[1] * f))
                    Pm += Pq
                    Am += Aq
                P, A = Pm / subs, Am / subs
            if defocus:
                k = disk_kernel(defocus)
                P = cv2.filter2D(P, -1, k, borderType=cv2.BORDER_CONSTANT)
                A = np.clip(cv2.filter2D(A, -1, k, borderType=cv2.BORDER_CONSTANT), 0, 1)
            Pg = P                      # GT: no screen light on the subject
            P = add_spill(P, A, B, spill)
            if smoke:
                Psm, Asm = smoke_layer(H, W, smoke_rng, t, smoke)
                P = Psm + (1 - Asm[..., None]) * P
                Pg = Psm + (1 - Asm[..., None]) * Pg
                A = Asm + (1 - Asm) * A
            P = P * exposure
            Pg = Pg * exposure
            I = P + (1 - A[..., None]) * (B * exposure)
            I = I + camera_noise(I, rng, iso)
            d = sharpen(lin_to_srgb(I), sharp)
            disp.append(np.clip(d, 0, 1).astype(np.float32))
            gt_a.append(A.astype(np.float32))
            gt_f.append(np.where(A[..., None] > 1e-4, Pg / np.maximum(A, 1e-4)[..., None], 0).astype(np.float32))

        disp = _encode_decode(disp, kind, crf, rng)
        to_t = lambda xs: torch.from_numpy(np.stack(xs))
        return {
            "display": to_t(disp).permute(0, 3, 1, 2).contiguous(),        # (T,3,H,W) sRGB [0,1]
            "alpha": to_t(gt_a).unsqueeze(1),                             # (T,1,H,W)
            "fg": to_t(gt_f).permute(0, 3, 1, 2).contiguous(),            # (T,3,H,W) linear
        }
