#!/usr/bin/env python3
"""Fine-tune CorridorKey 2.0 for real-camera fidelity.

What trains:
  * a new full-resolution-aware refinement head (the shipped but disabled
    `refine4` head: /4 ConvNeXt over pixel-unshuffled RGB plus stage-2
    tokens, zero-initialized so step 0 reproduces the released model);
  * all of stage 2;
  * the stage-1 /16 decoder and its heads (lower learning rate).
The frozen foundations and the stage-1 /64 body stay fixed, which keeps the
model's subject understanding and makes the run cheap.

Data comes from train/data.py (train splits only; the benchmark uses the
test splits). Output is a runtime checkpoint that infer.py loads with
--weights.

  python train/finetune.py --data ~/corridorkey/data --out ~/corridorkey/ft/run1 --steps 20000
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "train"))

from corridorkey import color_utils as cu  # noqa: E402
from corridorkey.model import CorridorKeyV3  # noqa: E402
from corridorkey.runtime import RuntimePaths, load_runtime_checkpoint, prepare_runtime_environment  # noqa: E402
from data import KeyingClips  # noqa: E402

TRAIN_STAGE1 = ("dec_16.", "up.", "uskip_scale", "patch_late_skip_scale", "aux_depatchify.")


def build_model(paths: RuntimePaths, refine4_blocks: int):
    prepare_runtime_environment(paths)
    ckpt = load_runtime_checkpoint(paths.weights)
    cfg = dict(ckpt["model_config"])
    base_cfg = dict(cfg)
    cfg.update(cradio_repo=str(paths.cradio_dir), moge_checkpoint=str(paths.moge_checkpoint),
               rvm_ckpt=str(paths.rvm_ckpt), foundation_stats_path=str(paths.foundation_stats),
               gradient_checkpoint=True, device="cuda", stage1_recurrent_64_sites=(),
               refine4_blocks=refine4_blocks, stage2_temporal_mix=True, stage2_key_tail=True)
    model = CorridorKeyV3(**cfg)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    new_parts = ("refine4", "temporal_mix", "key_tail")
    missing = [k for k in missing if not k.startswith("foundations.") and not any(p in k for p in new_parts)]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    base_cfg.update(refine4_blocks=refine4_blocks, stage2_temporal_mix=True, stage2_key_tail=True)
    model.stage1.cuda()
    model.stage2.cuda()
    return model, base_cfg, ckpt.get("metadata", {})


def trainable_groups(model, lr):
    new, stage2, stage1 = [], [], []
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        if name.startswith("foundations."):
            continue
        if any(k in name for k in ("refine4", "temporal_mix", "key_tail")):
            new.append(p)
        elif name.startswith("stage2."):
            stage2.append(p)
        elif name.startswith("stage1.") and name[len("stage1."):].startswith(TRAIN_STAGE1):
            stage1.append(p)
        else:
            continue
        p.requires_grad_(True)
    return [
        {"params": new, "lr": lr * 10, "base_lr": lr * 10},
        {"params": stage2, "lr": lr, "base_lr": lr},
        {"params": stage1, "lr": lr * 0.5, "base_lr": lr * 0.5},
    ]


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

_GK = None


def _gauss(x):
    global _GK
    if _GK is None or _GK.device != x.device:
        g = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=x.device)
        k = (g[:, None] * g[None, :]) / 256
        _GK = k.view(1, 1, 5, 5)
    c = x.shape[1]
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), _GK.expand(c, 1, 5, 5), groups=c)


def laplacian_loss(pred, gt, levels=5):
    loss = 0.0
    p, g = pred, gt
    for i in range(levels):
        pd = F.avg_pool2d(_gauss(p), 2)
        gd = F.avg_pool2d(_gauss(g), 2)
        pl = p - F.interpolate(pd, size=p.shape[-2:], mode="bilinear", align_corners=False)
        gl = g - F.interpolate(gd, size=g.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + (2 ** i) * (pl - gl).abs().mean()
        p, g = pd, gd
    return loss + (2 ** levels) * (p - g).abs().mean()


def random_backgrounds(n, H, W, device, gen):
    flat = torch.rand(n, 3, 1, 1, device=device, generator=gen) ** 2.2
    tex = torch.rand(n, 3, H // 32 + 1, W // 32 + 1, device=device, generator=gen) ** 2.2
    tex = F.interpolate(tex, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
    mix = (torch.rand(n, 1, 1, 1, device=device, generator=gen) < 0.5).float()
    return mix * flat.expand(-1, -1, H, W) + (1 - mix) * tex


def keying_loss(alpha, fg_asinh, gt_a, gt_f, bgs):
    """alpha (N,1,H,W) raw, fg_asinh (N,3,H,W), gt_a (N,1,H,W), gt_f linear (N,3,H,W)."""
    if alpha.shape[-2:] != gt_a.shape[-2:]:
        gt_f = F.interpolate(gt_f * gt_a, size=alpha.shape[-2:], mode="area")
        gt_a = F.interpolate(gt_a, size=alpha.shape[-2:], mode="area")
        gt_f = gt_f / gt_a.clamp(min=1e-4)
        bgs = F.interpolate(bgs, size=alpha.shape[-2:], mode="area")
    l_a = (alpha - gt_a).abs().mean() + laplacian_loss(alpha, gt_a)
    w = gt_a * (gt_a > 0.01)
    l_f = ((fg_asinh - cu.linear_to_asinh(gt_f)).abs() * w).sum() / w.sum().clamp(min=1.0) / 3
    a = alpha.clamp(0, 1)
    f_lin = cu.asinh_to_linear(fg_asinh)
    comp = cu.linear_to_srgb((a * f_lin + (1 - a) * bgs).clamp(min=0))
    comp_gt = cu.linear_to_srgb((gt_a * gt_f + (1 - gt_a) * bgs).clamp(min=0))
    l_c = (comp - comp_gt).abs().mean()
    return l_a, l_f, l_c


def temporal_loss(alpha, gt_a, T):
    a = alpha.view(-1, T, *alpha.shape[1:])
    g = gt_a.view(-1, T, *gt_a.shape[1:])
    return ((a[:, 1:] - a[:, :-1]) - (g[:, 1:] - g[:, :-1])).abs().mean()


def degrade_hint(gt_a0, gen):
    """A rough artist/auto matte from the GT: blur, grow/shrink, threshold."""
    h = gt_a0
    r = int(torch.randint(1, 12, (1,), generator=gen).item())
    h = F.avg_pool2d(h, 2 * r + 1, stride=1, padding=r)
    thr = float(torch.empty(1).uniform_(0.2, 0.8, generator=gen))
    return (h > thr).float()


# --------------------------------------------------------------------------


def local_sgd_sync(named, ema, args, rnd):
    """Average trainable weights (and their EMA) with the other workers."""
    d = args.sync_dir
    d.mkdir(parents=True, exist_ok=True)
    mine = {n: p.detach().to("cpu", torch.float16) for n, p in named.items()}
    tmp = d / f".w{args.rank}_r{rnd}.pt"
    torch.save({"w": mine, "ema": {n: v.to("cpu", torch.float16) for n, v in ema.items()}}, tmp)
    tmp.rename(d / f"w{args.rank}_r{rnd}.pt")
    avg = d / f"avg_r{rnd}.pt"
    t0 = time.time()
    while not avg.exists() and time.time() - t0 < args.sync_timeout:
        time.sleep(5)
    if not avg.exists():
        print(f"sync round {rnd}: no average after {args.sync_timeout:.0f}s, continuing alone", flush=True)
        return
    st = torch.load(avg, map_location="cuda")
    with torch.no_grad():
        for n, p in named.items():
            p.copy_(st["w"][n].to(p.dtype))
            ema[n].copy_(st["ema"][n].float())
    print(f"sync round {rnd}: averaged over {st.get('n', '?')} workers", flush=True)


def save_runtime(model, ema, cfg, metadata, path: Path):
    from safetensors.torch import save_file
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("foundations."):
            continue
        sd[k] = (ema[k] if k in ema else v).detach().float().cpu().contiguous()
    meta = dict(metadata)
    meta["public_model_name"] = "CorridorKey 3.0 (unofficial community fine-tune of CorridorKey 2.0)"
    save_file(sd, str(path), metadata={
        "format": "corridorkey_v2_runtime_checkpoint",
        "model_config": json.dumps(cfg),
        "metadata": json.dumps(meta),
        "format_version": "1",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--crop", type=int, nargs=2, default=(512, 896))
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--refine4_blocks", type=int, default=2)
    ap.add_argument("--hint_prob", type=float, default=0.3)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", type=Path, default=None, help="start from these runtime weights instead of 2.0")
    # local SGD across machines: every sync_every steps each worker writes its
    # trainable weights to sync_dir/w<rank>_r<round>.pt and waits for
    # sync_dir/avg_r<round>.pt from the coordinator (bench/../train/sync.py).
    ap.add_argument("--sync_dir", type=Path, default=None)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--sync_every", type=int, default=600)
    ap.add_argument("--sync_timeout", type=float, default=1800)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    paths = RuntimePaths.from_package(REPO, str(args.init) if args.init else None)
    model, cfg, metadata = build_model(paths, args.refine4_blocks)
    model.stage1.precise_io = True
    model.stage2.precise_io = True
    groups = trainable_groups(model, args.lr)
    params = [p for g in groups for p in g["params"]]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M "
          f"(new {sum(p.numel() for p in groups[0]['params']) / 1e6:.2f}M)", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.99))
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    ema = {n: p.detach().clone().float() for n, p in named.items()}
    step = 0
    state_path = args.out / "state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        ema = st["ema"]
        step = st["step"]
        print(f"resumed at step {step}", flush=True)

    ds = KeyingClips(args.data, crop_hw=tuple(args.crop), T=args.T, seed=step + 1)
    dl = DataLoader(ds, batch_size=1, num_workers=args.workers, prefetch_factor=4,
                    persistent_workers=True, pin_memory=True)
    gen = torch.Generator(device="cuda").manual_seed(1234 + step)
    cgen = torch.Generator().manual_seed(4321 + step)
    model.train()
    log = open(args.out / "train.log", "a")
    t_last = time.time()
    run = {}
    it = iter(dl)
    while step < args.steps:
        warm = min(1.0, (step + 1) / 300)
        cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * warm * cos
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            batch = next(it)
            disp = batch["display"].cuda(non_blocking=True)        # (1,T,3,H,W)
            gt_a = batch["alpha"].cuda(non_blocking=True)
            gt_f = batch["fg"].cuda(non_blocking=True)
            B, T, _, H, W = disp.shape
            rgb = cu.linear_to_asinh(cu.srgb_to_linear(disp))
            flat = disp.reshape(B * T, 3, H, W)
            rgb_cradio = F.interpolate(flat, size=(H // 4, W // 4), mode="area").view(B, T, 3, H // 4, W // 4)
            rgb_moge = F.interpolate(flat, size=(7 * H // 32, 7 * W // 32), mode="area").view(
                B, T, 3, 7 * H // 32, 7 * W // 32)
            hint = torch.zeros(B, T, 1, H, W, device="cuda")
            hq = torch.full((B, T), -1.0, device="cuda")
            if torch.rand(1, generator=cgen).item() < args.hint_prob:
                hint[:, 0] = degrade_hint(gt_a[:, 0], cgen)
                hq.zero_()
                hq[:, 0] = float(torch.empty(1).uniform_(0.5, 1.0, generator=cgen))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(rgb, hint, rgb_cradio, rgb_moge, hint_quality=hq)
            ga = gt_a.reshape(B * T, 1, H, W)
            gf = gt_f.reshape(B * T, 3, H, W)
            bgs = random_backgrounds(B * T, H, W, "cuda", gen)
            a = out["alpha"].float().reshape(B * T, 1, H, W)
            f = out["fg"].float().reshape(B * T, 3, H, W)
            la, lf, lc = keying_loss(a, f, ga, gf, bgs)
            lt = temporal_loss(a, ga, T)
            ab = out["alpha_base"].float().reshape(B * T, 1, H, W)
            fb = out["fg_base"].float().reshape(B * T, 3, H, W)
            la_b, lf_b, lc_b = keying_loss(ab, fb, ga, gf, bgs)
            ax = out["alpha_aux"].float().reshape(B * T, 1, H // 2, W // 2)
            fx = out["fg_aux"].float().reshape(B * T, 3, H // 2, W // 2)
            la_x, lf_x, lc_x = keying_loss(ax, fx, ga, gf, bgs)
            aux = model.stage2.tail_aux
            w_fg = ga * (ga > 0.01)
            lf_net = ((cu.linear_to_asinh(aux["f_net"]) - cu.linear_to_asinh(gf)).abs() * w_fg).sum() \
                / w_fg.sum().clamp(min=1.0) / 3
            loss = (la + lf + 2 * lc + 0.5 * lt + 0.25 * lf_net
                     + 0.25 * (la_b + lf_b + lc_b) + 0.25 * (la_x + lf_x + lc_x))
            (loss / args.accum).backward()
            for k, v in (("alpha", la), ("fg", lf), ("comp", lc), ("temporal", lt), ("total", loss)):
                run[k] = run.get(k, 0.0) + float(v) / args.accum
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for n, p in named.items():
                ema[n].mul_(d).add_(p.detach().float(), alpha=1 - d)
        if args.sync_dir is not None and step % args.sync_every == 0:
            local_sgd_sync(named, ema, args, step // args.sync_every)
        if step % 50 == 0:
            dt = (time.time() - t_last) / 50
            t_last = time.time()
            msg = (f"step {step} " + " ".join(f"{k}={v / 50:.4f}" for k, v in run.items())
                   + f" gnorm={float(gn):.3f} lr={opt.param_groups[1]['lr']:.2e} {dt:.2f}s/step")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            run = {}
        if step % args.save_every == 0 or step == args.steps:
            save_runtime(model, ema, cfg, metadata, args.out / f"corridorkey_ft_{step:06d}.safetensors")
            torch.save({"model": {n: p.detach() for n, p in named.items()}, "opt": opt.state_dict(),
                        "ema": ema, "step": step}, state_path)
            print(f"saved step {step}", flush=True)


if __name__ == "__main__":
    main()
