#!/usr/bin/env python3
"""Build the CorridorKey hard benchmark.

Every case composites a real foreground with an exact ground-truth alpha over a
simulated keying stage, in scene-linear light, then degrades the plate the way
cameras and delivery codecs do. Because the plate is constructed, the ground
truth is exact for every frame:

    plate   = encode(camera(alpha * F + (1 - alpha) * B))
    gt      = alpha, F (+ the sensor noise that landed on the subject)

Sensor noise is part of the subject as captured, so it is kept in the GT
foreground: a compositor wants the plate's own grain on the subject. Codec and
sharpening damage are not part of the GT.

Sources (held-out test splits only, so the train splits stay free for
fine-tuning):
  vm    VideoMatte240K HD test clips (real motion, real hair)
  aim   AIM-500 (fine detail and transparent categories)
  t460  Transparent-460 test (glass, smoke, liquids)

Usage:
  python bench/make_bench.py --data ~/corridorkey/data --out ~/corridorkey/bench
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
import shutil
import subprocess
import zlib
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

FPS = 24


# ---------------------------------------------------------------------------
# color
# ---------------------------------------------------------------------------

def srgb_to_lin(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def luma(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


# ---------------------------------------------------------------------------
# foreground sources
# ---------------------------------------------------------------------------

def _read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    img = img.astype(np.float32) / scale
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)


def _read_alpha(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    a = a.astype(np.float32) / scale
    if a.ndim == 3:
        # Some alpha PNGs are RGBA with a constant alpha channel and the matte in RGB.
        if a.shape[2] == 4 and a[..., 3].min() < a[..., 3].max():
            a = a[..., 3]
        else:
            a = a[..., 0]
    return a


class VideoMatteClip:
    """VideoMatte240K_JPEG_HD/test/{fgr,pha}/<clip>/<frame>.jpg"""

    def __init__(self, root: Path, clip: str):
        self.fgr = sorted((root / "fgr" / clip).iterdir())
        self.pha = sorted((root / "pha" / clip).iterdir())
        assert len(self.fgr) == len(self.pha) and self.fgr, clip

    def __len__(self):
        return len(self.fgr)

    def frame(self, i):
        i = min(i, len(self.fgr) - 1)
        return srgb_to_lin(_read_rgb(self.fgr[i])), _read_alpha(self.pha[i])


class Still:
    def __init__(self, fg_srgb, alpha):
        self.fg = srgb_to_lin(fg_srgb)
        self.alpha = alpha

    def __len__(self):
        return 10 ** 9

    def frame(self, i):
        return self.fg, self.alpha


def load_aim(data: Path, idx: int) -> Still:
    import pyarrow.parquet as pq
    table = pq.read_table(data / "aim500.parquet")
    row = table.slice(idx, 1).to_pylist()[0]
    imgs = {}
    for k, v in row.items():
        if isinstance(v, dict) and "bytes" in v:
            arr = cv2.imdecode(np.frombuffer(v["bytes"], np.uint8), cv2.IMREAD_UNCHANGED)
            imgs[k] = arr
    img_key = next(k for k in imgs if imgs[k].ndim == 3 and imgs[k].shape[2] >= 3)
    a_key = next(k for k in imgs if k != img_key)
    img = cv2.cvtColor(imgs[img_key][..., :3], cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    a = imgs[a_key]
    a = (a[..., 0] if a.ndim == 3 else a).astype(np.float64) / (65535.0 if a.dtype == np.uint16 else 255.0)
    # AIM-500 ships the original photo, whose edge pixels still hold the old
    # background. Recover a clean foreground (Germer et al. multilevel solve).
    from pymatting import estimate_foreground_ml
    fg = estimate_foreground_ml(img, a).astype(np.float32)
    return Still(np.clip(fg, 0, 1), a.astype(np.float32))


def load_t460(data: Path, name: str) -> Still:
    root = data / "t460" / "Test"
    fg = _read_rgb(root / "fg" / f"{name}.jpg")
    a = _read_alpha(root / "alpha" / f"{name}.png")
    return Still(fg, a)


# ---------------------------------------------------------------------------
# layer placement and motion
# ---------------------------------------------------------------------------

def place(fg_lin, alpha, H, W, height_frac=None, center=(0.5, 0.5), overscan=1.0):
    """Fit a foreground layer onto an HxW canvas in premultiplied space.

    height_frac=None fills the canvas like a full frame (cover-crop);
    otherwise the alpha bbox is scaled to height_frac * H at `center`.
    """
    premult = fg_lin * alpha[..., None]
    h, w = alpha.shape
    if height_frac is None:
        s = max(H / h, W / w) * overscan
        cy, cx = h / 2, w / 2
        ty, tx = H / 2, W / 2
    else:
        ys, xs = np.nonzero(alpha > 0.02)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        s = height_frac * H / max(y1 - y0, 1)
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        ty, tx = center[1] * H, center[0] * W
    M = np.array([[s, 0, tx - s * cx], [0, s, ty - s * cy]], np.float32)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    if s < 1:
        # INTER_AREA is only honored by resize, so resize first then shift.
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        premult = cv2.resize(premult, (nw, nh), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
        M = np.array([[1, 0, tx - s * cx], [0, 1, ty - s * cy]], np.float32)
        interp = cv2.INTER_LINEAR
    P = cv2.warpAffine(premult, M, (W, H), flags=interp, borderValue=0)
    A = cv2.warpAffine(alpha, M, (W, H), flags=interp, borderValue=0)
    A = np.clip(A, 0.0, 1.0)
    P = np.clip(P, 0.0, None)
    return P, A


def warp(P, A, M):
    H, W = A.shape
    P = cv2.warpAffine(P, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    A = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    return P, np.clip(A, 0.0, 1.0)


def affine(tx=0.0, ty=0.0, rot_deg=0.0, scale=1.0, cx=0.0, cy=0.0):
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, scale).astype(np.float32)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def disk_kernel(r):
    r = float(r)
    n = int(np.ceil(r)) * 2 + 1
    yy, xx = np.mgrid[:n, :n] - n // 2
    d = np.sqrt(xx ** 2 + yy ** 2)
    k = np.clip(r + 0.5 - d, 0, 1).astype(np.float32)
    return k / k.sum()


# ---------------------------------------------------------------------------
# stage and camera
# ---------------------------------------------------------------------------

def smooth_noise(H, W, cell, rng):
    gh, gw = int(np.ceil(H / cell)) + 3, int(np.ceil(W / cell)) + 3
    g = rng.standard_normal((gh, gw)).astype(np.float32)
    up = cv2.resize(g, (gw * cell, gh * cell), interpolation=cv2.INTER_CUBIC)
    return up[cell:cell + H, cell:cell + W]


def fbm(H, W, rng, base_cell, octaves=5):
    out = np.zeros((H, W), np.float32)
    amp, total, cell = 1.0, 0.0, base_cell
    for _ in range(octaves):
        out += amp * smooth_noise(H, W, max(cell, 2), rng)
        total += amp
        amp *= 0.5
        cell = max(cell // 2, 2)
    return out / total


SCREEN = {
    "green": np.array([0.045, 0.32, 0.065], np.float32),
    "blue": np.array([0.03, 0.075, 0.36], np.float32),
}


def make_backing(H, W, rng, screen="green", uneven=0.0, wrinkles=0.0,
                 markers=0, vignette=0.0):
    B = np.ones((H, W, 3), np.float32) * SCREEN[screen]
    gain = np.ones((H, W), np.float32)
    if uneven:
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        ramp = (xx / W - 0.5) * rng.uniform(-1, 1) + (yy / H - 0.5) * rng.uniform(-1, 1)
        gain *= 1.0 + uneven * (0.6 * smooth_noise(H, W, max(H // 3, 8), rng) + ramp)
    if wrinkles:
        n = rng.standard_normal((H, W)).astype(np.float32)
        n = ndimage.gaussian_filter(n, sigma=(10, 90))
        n /= n.std() + 1e-6
        n = ndimage.rotate(n, rng.uniform(-35, 35), reshape=False, mode="reflect")
        gain *= 1.0 + wrinkles * n
    if vignette:
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        gain *= 1.0 - vignette * np.clip(r2, 0, 2) / 2
    B *= np.clip(gain, 0.2, 2.0)[..., None]
    if markers:
        mark = np.zeros((H, W), np.float32)
        arm = max(H // 45, 6)
        th = max(H // 300, 2)
        for _ in range(markers):
            x, y = int(rng.uniform(0.05, 0.95) * W), int(rng.uniform(0.05, 0.95) * H)
            cv2.line(mark, (x - arm, y), (x + arm, y), 1.0, th, cv2.LINE_AA)
            cv2.line(mark, (x, y - arm), (x, y + arm), 1.0, th, cv2.LINE_AA)
        tape = np.array([0.22, 0.24, 0.23], np.float32)
        B = B * (1 - mark[..., None]) + tape * mark[..., None]
    return B


def add_spill(P, A, B, strength):
    """Bounce light from the screen onto the subject, strongest near edges.

    This is real light on the subject, so it stays in the GT foreground."""
    if strength <= 0:
        return P
    near_edge = ndimage.gaussian_filter(1.0 - A, sigma=max(A.shape[0] / 60, 4))
    w = (0.25 + 0.75 * np.clip(near_edge * 2.5, 0, 1)) * A
    Bb = cv2.blur(B, (61, 61))
    return P + strength * Bb * w[..., None]


def green_wardrobe(P, A, strength=0.85):
    """Recolor the lower half of the subject toward a screen-adjacent green."""
    ys = np.nonzero(A.max(axis=1) > 0.5)[0]
    if ys.size == 0:
        return P
    y0, y1 = ys.min(), ys.max()
    H = A.shape[0]
    yy = np.arange(H, dtype=np.float32)[:, None]
    m = np.clip((yy - (y0 + 0.5 * (y1 - y0))) / (0.08 * (y1 - y0) + 1), 0, 1)
    m = (m * (A > 0.98)).astype(np.float32)
    m = ndimage.gaussian_filter(m, 3) * A
    Fl = luma(P / np.maximum(A, 1e-4)[..., None])
    green = Fl[..., None] * np.array([0.55, 1.35, 0.6], np.float32)
    F = P / np.maximum(A, 1e-4)[..., None]
    F2 = F * (1 - strength * m[..., None]) + green * strength * m[..., None]
    return F2 * A[..., None]


def smoke_layer(H, W, rng, t, density=0.7):
    """Procedural drifting smoke: (premultiplied color, alpha)."""
    if not hasattr(smoke_layer, "_cache") or smoke_layer._cache[0] != (H, W):
        big = fbm(H * 2, W, rng, base_cell=max(H // 4, 8), octaves=6)
        big2 = fbm(H * 2, W, rng, base_cell=max(H // 7, 8), octaves=5)
        smoke_layer._cache = ((H, W), big, big2)
    _, big, big2 = smoke_layer._cache
    o1 = int(t * H * 0.012) % H
    o2 = int(t * H * 0.02) % H
    n = 0.6 * big[H - o1:2 * H - o1] + 0.4 * big2[H - o2:2 * H - o2]
    n = (n - n.mean()) / (n.std() + 1e-6)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    a = np.clip((n * 0.35 + 0.15) * (0.4 + 0.9 * yy), 0, 1) * density
    a = np.clip(a, 0, 0.85).astype(np.float32)
    color = np.array([0.42, 0.40, 0.38], np.float32)
    return color * a[..., None], a


def camera_noise(I, rng, iso):
    """Poisson-Gaussian sensor noise in linear light, lightly demosaic-correlated."""
    if iso <= 0:
        return np.zeros_like(I)
    a = 0.0009 * iso
    b = (0.0012 * iso) ** 2
    sigma = np.sqrt(np.clip(a * np.clip(I, 0, None) + b, 0, None))
    n = rng.standard_normal(I.shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.6) * 1.6
    return (n * sigma).astype(np.float32)


def sharpen(display, amount):
    if amount <= 0:
        return display
    blur = cv2.GaussianBlur(display, (0, 0), 1.4)
    return display + amount * (display - blur)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def _ffmpeg_rawvideo_in(W, H):
    return ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb48le", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]


_TAGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
_TO709 = ["-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int"]


def encode(display_frames, linear_frames, out_dir: Path, kind: str, crf: int = 18):
    H, W = display_frames[0].shape[:2]
    if kind == "png16":
        d = out_dir / "plate_png"
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(display_frames):
            u16 = (np.clip(f, 0, 1) * 65535 + 0.5).astype(np.uint16)
            cv2.imwrite(str(d / f"{i:05d}.png"), cv2.cvtColor(u16, cv2.COLOR_RGB2BGR))
        return d.name
    if kind == "exr_linear":
        d = out_dir / "plate_exr"
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(linear_frames):
            cv2.imwrite(str(d / f"{i:05d}.exr"), cv2.cvtColor(f.astype(np.float32), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        return d.name
    if kind == "prores10":
        name = "plate.mov"
        codec = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    elif kind == "h264_8":
        name = "plate.mp4"
        codec = ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                 "-pix_fmt", "yuv420p"]
    else:
        raise ValueError(kind)
    cmd = _ffmpeg_rawvideo_in(W, H) + _TO709 + codec + _TAGS + [str(out_dir / name)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in display_frames:
        u16 = (np.clip(f, 0, 1) * 65535 + 0.5).astype("<u2")
        proc.stdin.write(u16.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_dir / name}")
    return name


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

# Each case: what it stresses, and the knobs that make it hard. Defaults are a
# clean, evenly lit green screen shot losslessly.
CASES = [
    dict(id="c01_clean_control", why="Lossless, even screen, no spill or noise. The ceiling.",
         src=("vm", 0, 0), fmt="png16"),
    dict(id="c02_real_stage", why="Uneven screen, wrinkles, tracking markers, edge spill, ProRes 10-bit.",
         src=("vm", 1, 0), fmt="prores10", uneven=0.45, wrinkles=0.06, markers=10,
         vignette=0.35, spill=0.22, iso=1.0),
    dict(id="c03_camera_h264", why="Real-stage plate through a camera-grade 8-bit 4:2:0 H.264.",
         src=("vm", 2, 0), fmt="h264_8", crf=20, uneven=0.45, wrinkles=0.06, markers=10,
         vignette=0.35, spill=0.22, iso=2.5),
    dict(id="c04_crushed_web", why="Heavily compressed 8-bit with in-camera sharpening halos.",
         src=("vm", 3, 0), fmt="h264_8", crf=31, uneven=0.35, markers=6, spill=0.15,
         iso=3.0, sharpen=0.8),
    dict(id="c05_motion_blur", why="Fast whip move, 180-degree shutter, exact integrated blur.",
         src=("vm", 4, 0), fmt="prores10", uneven=0.3, spill=0.15, iso=1.0, motion="whip"),
    dict(id="c06_focus_pull", why="Defocus ramps from sharp to 14 px bokeh across the shot.",
         src=("vm", 0, 60), fmt="prores10", uneven=0.3, spill=0.15, iso=1.0, defocus=14.0),
    dict(id="c07_smoke_veil", why="Drifting semi-transparent smoke over the subject.",
         src=("vm", 1, 60), fmt="prores10", uneven=0.3, spill=0.1, iso=1.0, smoke=0.75),
    dict(id="c08_blue_screen", why="Blue backing with blue spill (green-only despill heuristics fail).",
         src=("vm", 2, 60), fmt="prores10", screen="blue", uneven=0.45, wrinkles=0.05,
         markers=8, vignette=0.3, spill=0.25, iso=1.0),
    dict(id="c09_low_light", why="Screen 2.5 stops under, heavy noise, 8-bit H.264.",
         src=("vm", 3, 60), fmt="h264_8", crf=18, exposure=-2.5, uneven=0.5, spill=0.2,
         iso=6.0),
    dict(id="c10_green_wardrobe", why="Lower body dressed in a screen-adjacent green.",
         src=("vm", 4, 60), fmt="prores10", uneven=0.3, spill=0.15, iso=1.0, wardrobe=0.85),
    dict(id="c11_glass_transparency", why="Glass splash and water (Transparent-460), slow drift.",
         src=("t460", 2, 0), fmt="prores10", uneven=0.35, markers=6, iso=1.0,
         height_frac=0.75, drift=True),
    dict(id="c12_spiderweb_4k", why="UHD 3840x2160 spider web: sub-pixel threads (Transparent-460).",
         src=("t460", 46, 0), fmt="prores10", res=(2160, 3840), frames=24, uneven=0.35,
         markers=6, iso=1.0, drift=True),
    dict(id="c13_hdr_linear", why="Scene-linear EXR with highlights above 1.0 (HDR path).",
         src=("vm", 0, 120), fmt="exr_linear", uneven=0.3, spill=0.15, iso=1.0,
         fg_gain=3.0),
    dict(id="c14_animal_fur", why="Animal fur (AIM-500, foreground re-estimated), slow drift.",
         src=("aim", 0, 0), fmt="prores10", uneven=0.35, wrinkles=0.05, markers=6, spill=0.2,
         iso=1.5, drift=True),
]


def resolve_source(spec, data: Path, pick):
    kind, idx, start = spec
    if kind == "vm":
        root = data / "VideoMatte240K_JPEG_HD" / "test"
        clips = sorted(p.name for p in (root / "fgr").iterdir())
        return VideoMatteClip(root, clips[idx % len(clips)]), start
    if kind == "aim":
        return load_aim(data, pick["aim"]), 0
    if kind == "t460":
        names = sorted(p.stem for p in (data / "t460" / "Test" / "fg").iterdir())
        return load_t460(data, names[idx % len(names)]), 0
    raise ValueError(kind)


def build_case(case, data: Path, out: Path, frames: int, pick: dict):
    rng = np.random.default_rng(zlib.crc32(case["id"].encode()))
    H, W = case.get("res", (1080, 1920))
    T = case.get("frames", frames)
    src, start = resolve_source(case["src"], data, pick)
    d = out / case["id"]
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

    exposure = 2.0 ** case.get("exposure", 0.0)
    B = make_backing(H, W, rng, case.get("screen", "green"), case.get("uneven", 0.0),
                     case.get("wrinkles", 0.0), case.get("markers", 0),
                     case.get("vignette", 0.0))
    smoke_rng = np.random.default_rng(7)

    gt_a = np.zeros((T, H, W), np.float16)
    gt_f = np.zeros((T, H, W, 3), np.float16)
    displays, linears = [], []
    for t in range(T):
        fg, a = src.frame(start + t)
        P0, A0 = place(fg, a, H, W, case.get("height_frac"),
                       overscan=1.1 if case.get("drift") and not case.get("height_frac") else 1.0)
        if case.get("fg_gain"):
            # HDR: lift the subject's highlights well above 1.0
            Fl = P0 / np.maximum(A0, 1e-4)[..., None]
            hi = np.clip((luma(Fl) - 0.25) / 0.5, 0, 1)[..., None]
            P0 = P0 * (1 + (case["fg_gain"] - 1) * hi)
        if case.get("wardrobe"):
            P0 = green_wardrobe(P0, A0, case["wardrobe"])

        if case.get("motion") == "whip":
            # Exact motion blur: average premultiplied layers over the open shutter.
            subs = 16
            P = np.zeros_like(P0)
            A = np.zeros_like(A0)
            for s in range(subs):
                tt = t + 0.5 * (s + 0.5) / subs   # 180-degree shutter
                # speed = v * sin^2(pi tt / T): still, whip through mid-shot, settle
                v = W * 0.03
                x = v * (tt / 2 - T / (4 * np.pi) * np.sin(2 * np.pi * tt / T)) - W * 0.3
                Ps, As = warp(P0, A0, affine(tx=x, ty=8 * np.sin(tt * 0.7),
                                             rot_deg=2.5 * np.sin(tt * 0.4),
                                             cx=W / 2, cy=H / 2))
                P += Ps
                A += As
            P /= subs
            A /= subs
        elif case.get("drift"):
            P, A = warp(P0, A0, affine(tx=W * 0.0015 * t, ty=H * 0.002 * np.sin(t / 6),
                                       rot_deg=0.06 * t, cx=W / 2, cy=H / 2))
        else:
            P, A = P0, A0

        if case.get("defocus"):
            r = case["defocus"] * t / max(T - 1, 1)
            if r >= 0.5:
                k = disk_kernel(r)
                P = cv2.filter2D(P, -1, k, borderType=cv2.BORDER_CONSTANT)
                A = np.clip(cv2.filter2D(A, -1, k, borderType=cv2.BORDER_CONSTANT), 0, 1)

        P = add_spill(P, A, B, case.get("spill", 0.0))

        if case.get("smoke"):
            Ps, As = smoke_layer(H, W, smoke_rng, t, case["smoke"])
            P = Ps + (1 - As[..., None]) * P
            A = As + (1 - As) * A

        P = P * exposure
        Bt = B * exposure
        I = P + (1 - A[..., None]) * Bt
        n = camera_noise(I, rng, case.get("iso", 0.0))
        I_cam = I + n

        F = np.where(A[..., None] > 1e-4, (P + A[..., None] * n) / np.maximum(A, 1e-4)[..., None], 0)
        gt_a[t] = A
        gt_f[t] = np.clip(F, -65000, 65000)

        if case["fmt"] == "exr_linear":
            linears.append(I_cam)
            displays.append(None)
        else:
            disp = lin_to_srgb(I_cam)
            disp = sharpen(disp, case.get("sharpen", 0.0))
            displays.append(np.clip(disp, 0, 1))
        print(f"  {case['id']} frame {t + 1}/{T}", end="\r", flush=True)

    plate = encode(displays if displays[0] is not None else [np.zeros((H, W, 3))],
                   linears, d, case["fmt"], case.get("crf", 18))
    np.save(d / "gt_alpha.npy", gt_a)
    np.save(d / "gt_fg_linear.npy", gt_f)
    np.save(d / "backing_linear.npy", (B * exposure).astype(np.float16))
    meta = {k: v for k, v in case.items() if k != "src"}
    meta.update(src=list(case["src"]), plate=plate, frames=T, height=H, width=W, fps=FPS,
                linear_input=case["fmt"] == "exr_linear",
                source_name=getattr(src, "fgr", [Path("still")])[0].parent.name)
    (d / "case.json").write_text(json.dumps(meta, indent=1))
    print(f"  {case['id']}: {T} frames {W}x{H} {case['fmt']} -> {plate}          ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--aim", type=int, default=495, help="AIM-500 row for the fur case")
    ap.add_argument("--frames", dest="frames", type=int, default=40)
    args = ap.parse_args()
    pick = {"aim": args.aim}
    args.out.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        if args.only and case["id"] not in args.only:
            continue
        build_case(case, args.data, args.out, args.frames, pick)


if __name__ == "__main__":
    main()
