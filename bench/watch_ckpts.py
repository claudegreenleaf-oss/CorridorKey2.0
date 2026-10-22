#!/usr/bin/env python3
"""Benchmark every training checkpoint as it lands; stop the run on a plateau.

Runs on the home box next to training. Each new corridorkey_ft_<step>.safetensors
in --ckpts is scored on the 1080p cases (the 4K cases do not fit beside the
training job on one A6000), with the same fix flags as the "all" variant, and a
row goes into <out>/progress.tsv. When the best mean comp_mae has not improved by
--min_gain over the last --patience checkpoints, or the last two checkpoints are
both more than --diverge worse than the best one, the watcher writes the stop file
(the Thunder watchdog then deletes the cloud worker), stops the home training
unit and restarts ComfyUI.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FLAGS = ["--decode", "ffmpeg", "--precision", "mixed", "--plate_fg"]
METRICS = ["comp_mae", "comp_edge", "fg_mae", "dtssd"]


def mean(scores, k):
    return sum(v[k] for v in scores.values()) / len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=Path.home() / "corridorkey/bench")
    ap.add_argument("--ckpts", type=Path, default=Path.home() / "corridorkey/ft/v3")
    ap.add_argument("--out", type=Path, default=Path.home() / "corridorkey/results_v3")
    ap.add_argument("--reference", type=Path, default=Path.home() / "corridorkey/results2/all/scores.json",
                    help="released 2.0 + fixes, scored on the current bench (results/ predates the rebuild)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_gain", type=float, default=0.01)
    ap.add_argument("--diverge", type=float, default=0.10)
    ap.add_argument("--stop_file", type=Path, default=Path.home() / "corridorkey/STOP_THUNDER")
    ap.add_argument("--train_unit", default="ck-train")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cases = sorted(p.name for p in a.bench.iterdir() if (p / "case.json").exists()
                   and json.loads((p / "case.json").read_text())["height"] <= 1500)
    ref = None
    if a.reference.exists():
        ref = {k: v for k, v in json.loads(a.reference.read_text()).items() if k in cases}
    prog = a.out / "progress.tsv"
    if not prog.exists():
        prog.write_text("step\t" + "\t".join(METRICS) + "\tvs_ref_comp_mae\n")
        if ref:
            with prog.open("a") as f:
                f.write("ref\t" + "\t".join(f"{mean(ref, k):.4f}" for k in METRICS) + "\t\n")
    while True:
        done = {l.split("\t")[0] for l in prog.read_text().splitlines()[1:]}
        todo = [p for p in sorted(a.ckpts.glob("corridorkey_ft_*.safetensors"))
                if str(int(p.stem.split("_")[-1])) not in done]
        if not todo:
            if subprocess.run(["systemctl", "--user", "is-active", "-q", a.train_unit]).returncode:
                print("training unit stopped and nothing left to score", flush=True)
                subprocess.run(["systemctl", "--user", "start", "comfyui.service"])
                return
            time.sleep(120)
            continue
        ck = todo[0]
        step = int(ck.stem.split("_")[-1])
        res = a.out / f"s{step:06d}"
        print(f"{time.strftime('%F %T')} scoring step {step}", flush=True)
        r = subprocess.run([sys.executable, str(REPO / "bench/run_bench.py"), "--bench", str(a.bench),
                            "--out", str(res), "--skip_existing", "--only", *cases, "--",
                            "--weights", str(ck), *FLAGS], cwd=REPO)
        if r.returncode:
            print(f"step {step}: benchmark failed, retrying in 10 min", flush=True)
            time.sleep(600)
            continue
        scores = {k: v for k, v in json.loads((res / "scores.json").read_text()).items() if k in cases}
        cm = mean(scores, "comp_mae")
        vs = f"{(cm / mean(ref, 'comp_mae') - 1) * 100:+.1f}%" if ref else ""
        with prog.open("a") as f:
            f.write(f"{step}\t" + "\t".join(f"{mean(scores, k):.4f}" for k in METRICS) + f"\t{vs}\n")
        print(f"step {step}: comp_mae {cm:.4f} ({vs} vs released 2.0 + fixes)", flush=True)
        rows = [l.split("\t") for l in prog.read_text().splitlines()[1:] if l[:1].isdigit()]
        maes = [float(x[1]) for x in rows]
        best = rows[maes.index(min(maes))][0]
        print(f"best so far: step {best}", flush=True)
        reason = None
        if len(maes) >= 3 and min(maes[-2:]) > min(maes) * (1 + a.diverge):
            reason = f"diverging: last two checkpoints {a.diverge:.0%}+ worse than step {best}"
        elif len(maes) > a.patience and min(maes[-a.patience:]) > min(maes[:-a.patience]) * (1 - a.min_gain):
            reason = f"plateau: no {a.min_gain:.0%} gain over {a.patience} checkpoints"
        if reason:
            print(f"{reason}; stopping (best checkpoint: step {best})", flush=True)
            a.stop_file.touch()
            subprocess.run(["systemctl", "--user", "stop", a.train_unit])
            subprocess.run(["systemctl", "--user", "start", "comfyui.service"])
            return


if __name__ == "__main__":
    main()
