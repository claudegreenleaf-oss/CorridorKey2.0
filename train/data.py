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
    SCREEN, add_spill, affine, camera_noise, decontaminate_edges, disk_kernel, green_wardrobe,
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
    # Source mixture (renormalized over the sources that exist on disk).
    MIX = {"vm": 0.40, "render": 0.25, "p3m": 0.15, "am2k": 0.08, "t460": 0.12}

    def __init__(self, data_root: Path, crop_hw=(640, 1152), T=4, length=10 ** 7,
                 seed=0, mix=None):
        self.data_root = Path(data_root)
        self.crop_hw = crop_hw
        self.T = T
        self.length = length
        self.seed = seed
        d = self.data_root
        self.vm = _VMIndex(d / "VideoMatte240K_JPEG_HD" / "train")
        pairs = lambda fgdir, adir, aext: [
            (f, adir / (f.stem + aext)) for f in sorted(fgdir.glob("*.jpg"))
            if (adir / (f.stem + aext)).exists()] if fgdir.exists() else []
        self.stills = {
            "t460": pairs(d / "t460" / "Train" / "fg", d / "t460" / "Train" / "alpha", ".png"),
            "p3m": pairs(d / "p3m" / "fg", d / "p3m" / "alpha", ".png"),
            "am2k": pairs(d / "am2k" / "train" / "fg", d / "am2k" / "train" / "mask", ".png"),
        }
        self.renders_root = d / "renders"
        self._refresh_renders()
        mix = dict(mix or self.MIX)
        avail = {"vm": bool(self.vm.clips), **{k: len(v) > 0 for k, v in self.stills.items()},
                 "render": True}
        self.mix = {k: w for k, w in mix.items() if avail.get(k)}

    def _refresh_renders(self):
        """The render farm keeps adding clips; rescan now and then."""
        root = self.renders_root
        self.renders = sorted(p for p in root.glob("clip_*") if (p / "done").exists()
                              and (p / "fg_spill").exists()) if root.exists() else []

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
            a = (a0 if i == idx[0] else _read_alpha(pha[i]))[y0:y0 + ch, x0:x0 + cw]
            fg = decontaminate_edges(srgb_to_lin(_read_rgb(fgr[i])[y0:y0 + ch, x0:x0 + cw]), a)
            P = cv2.resize(fg * a[..., None], (W, H), interpolation=cv2.INTER_AREA)
            A = np.clip(cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA), 0, 1)
            if flip:
                P, A = P[:, ::-1].copy(), A[:, ::-1].copy()
            Ps.append(P)
            As.append(A)
        return Ps, As

    def _still_clip(self, rng, kind):
        H, W = self.crop_hw
        fg_path, a_path = self.stills[kind][rng.integers(len(self.stills[kind]))]
        fg = srgb_to_lin(_read_rgb(fg_path))
        a = _read_alpha(a_path)
        if rng.random() < 0.5:
            fg, a = fg[:, ::-1].copy(), a[:, ::-1].copy()
        h, w = a.shape
        s = float(rng.uniform(0.6, 1.3)) * min(H / h, W / w)
        P0 = cv2.resize(fg * a[..., None], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        A0 = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        canvasP = np.zeros((H, W, 3), np.float32)
        canvasA = np.zeros((H, W), np.float32)
        ph, pw = A0.shape
        oy, ox = (H - ph) // 2, (W - pw) // 2
        sy0, sx0 = max(0, -oy), max(0, -ox)
        dy0, dx0 = max(0, oy), max(0, ox)
        hh, ww = min(ph - sy0, H - dy0), min(pw - sx0, W - dx0)
        oy += int(rng.normal(0, H * 0.1))
        ox += int(rng.normal(0, W * 0.1))
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

    def _render_clip(self, rng):
        """A Blender clip: (plate foregrounds with spill, GT foregrounds, alphas, backings)."""
        H, W = self.crop_hw
        clip = self.renders[rng.integers(len(self.renders))]
        n = len(list((clip / "fg").glob("*.exr")))
        t0 = int(rng.integers(0, max(1, n - self.T + 1)))
        fr = [min(t0 + i, n - 1) + 1 for i in range(self.T)]
        rd = lambda p: cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt = [rd(clip / "fg" / f"{f:04d}.exr") for f in fr]
        sp = [rd(clip / "fg_spill" / f"{f:04d}.exr") for f in fr]
        bg = [rd(clip / "bg" / f"{f:04d}.exr")[..., :3] for f in fr]
        # The two foreground passes carry independent path-tracing noise
        # (about 15% per pixel at 128 samples), so fg alone is a target whose
        # noise the plate does not share. Screen spill is smooth, so take the
        # GT as the plate's foreground minus an alpha-normalized blur of the
        # spill: same mean, and exactly the plate's own noise.
        for g, p in zip(gt, sp):
            a = np.clip(g[..., 3], 0, 1)
            ab = cv2.GaussianBlur(a, (0, 0), 3)
            d = cv2.GaussianBlur(p[..., :3] - g[..., :3], (0, 0), 3) / np.maximum(ab, 1e-3)[..., None]
            g[..., :3] = np.clip(p[..., :3] - a[..., None] * d, 0, None)
        h, w = gt[0].shape[:2]
        s = float(rng.uniform(max(H / h, W / w), max(H / h, W / w) * 1.6))
        nh, nw = int(np.ceil(h * s)), int(np.ceil(w * s))
        y0, x0 = int(rng.integers(0, nh - H + 1)), int(rng.integers(0, nw - W + 1))
        flip = rng.random() < 0.5

        def fit(img):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            img = img[y0:y0 + H, x0:x0 + W]
            img = img[:, ::-1] if flip else img
            return np.ascontiguousarray(img)

        gt = [fit(x) for x in gt]
        sp = [fit(x) for x in sp]
        bg = [fit(x) for x in bg]
        bgr2rgb = lambda x: x[..., ::-1]
        A = [np.clip(x[..., 3], 0, 1) for x in gt]
        return ([bgr2rgb(x[..., :3]) for x in sp], [bgr2rgb(x[..., :3]) for x in gt], A,
                [bgr2rgb(x) for x in bg])

    # -- stage, camera, codec ------------------------------------------------
    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, os.getpid()))
        H, W = self.crop_hw
        T = self.T
        if i % 500 == 0:
            self._refresh_renders()
        mix = {k: w for k, w in self.mix.items() if k != "render" or len(self.renders) >= 20}
        keys = list(mix)
        src = keys[rng.choice(len(keys), p=np.array([mix[k] for k in keys]) / sum(mix.values()))]
        if src == "render":
            return self._camera_and_codec(rng, *self._render_clip(rng), src)
        Ps, As = self._vm_clip(rng) if src == "vm" else self._still_clip(rng, src)

        screen = "blue" if rng.random() < 0.2 else "green"
        B = make_backing(H, W, rng, screen,
                         uneven=float(rng.uniform(0, 0.55)),
                         wrinkles=float(rng.uniform(0, 0.07)) if rng.random() < 0.6 else 0.0,
                         markers=int(rng.integers(0, 10)) if rng.random() < 0.5 else 0,
                         vignette=float(rng.uniform(0, 0.4)))
        B *= float(np.exp(rng.normal(0, 0.25)))           # screen brightness / hue drift
        B *= (1 + rng.normal(0, 0.08, 3)).astype(np.float32)
        spill = float(rng.uniform(0, 0.3))
        motion = rng.random() < 0.25
        defocus = float(rng.uniform(2, 12)) if rng.random() < 0.15 else 0.0
        smoke = float(rng.uniform(0.3, 0.8)) if rng.random() < 0.12 else 0.0
        wardrobe = float(rng.uniform(0.5, 0.9)) if rng.random() < 0.1 else 0.0
        vel = rng.normal(0, W * 0.025, 2)
        smoke_rng = np.random.default_rng(int(rng.integers(1 << 31)))
        if hasattr(smoke_layer, "_cache"):
            del smoke_layer._cache

        Pp, Pgs, Al, Bl = [], [], [], []
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
            Pp.append(P)
            Pgs.append(Pg)
            Al.append(A)
            Bl.append(B)
        return self._camera_and_codec(rng, Pp, Pgs, Al, Bl, src)

    def _camera_and_codec(self, rng, Pp, Pgs, Al, Bl, src):
        """Exposure, sensor noise, sharpening and a real codec round trip."""
        exposure = 2.0 ** float(rng.uniform(-2.8, 0.8) if rng.random() < 0.5 else rng.normal(0, 0.3))
        iso = float(rng.choice([0.0, rng.uniform(0.3, 2.0), rng.uniform(2.0, 7.0)], p=[0.15, 0.55, 0.3]))
        kind = str(rng.choice(["png16", "prores10", "h264_8"], p=[0.15, 0.35, 0.5]))
        crf = int(rng.integers(14, 33))
        sharp = float(rng.uniform(0.3, 1.0)) if rng.random() < 0.25 else 0.0
        disp, gt_a, gt_f = [], [], []
        for P, Pg, A, B in zip(Pp, Pgs, Al, Bl):
            I = (P + (1 - A[..., None]) * B) * exposure
            I = I + camera_noise(I, rng, iso)
            d = sharpen(lin_to_srgb(I), sharp)
            disp.append(np.clip(d, 0, 1).astype(np.float32))
            gt_a.append(A.astype(np.float32))
            gt_f.append(np.where(A[..., None] > 1e-4, Pg * exposure / np.maximum(A, 1e-4)[..., None], 0)
                        .astype(np.float32))
        disp = _encode_decode(disp, kind, crf, rng)
        to_t = lambda xs: torch.from_numpy(np.stack(xs))
        return {
            "display": to_t(disp).permute(0, 3, 1, 2).contiguous(),        # (T,3,H,W) sRGB [0,1]
            "alpha": to_t(gt_a).unsqueeze(1),                             # (T,1,H,W)
            "fg": to_t(gt_f).permute(0, 3, 1, 2).contiguous(),            # (T,3,H,W) linear
        }
