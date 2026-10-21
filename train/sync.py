#!/usr/bin/env python3
"""Local-SGD coordinator for training across machines (runs next to worker 0).

Each round r, worker k writes sync_dir/w<k>_r<r>.pt and waits for
sync_dir/avg_r<r>.pt. This coordinator waits for the local worker's file,
pulls the same round from every remote listed in remotes.json (re-read each
round, so workers can join or leave mid-run), averages what arrived within
the grace window, writes avg_r<r>.pt and pushes it to the remotes.

remotes.json: [{"host": "user@host", "dir": "/remote/sync/dir", "rank": 1, "ssh": ["-p", "22"]}]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch


def scp(src, dst, ssh_opts):
    opts = []
    if "-p" in ssh_opts:
        opts = ["-P", ssh_opts[ssh_opts.index("-p") + 1]]
    extra = [o for o in ssh_opts if o not in ("-p",) and not o.isdigit()]
    return subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", *opts, *extra, src, dst],
                          capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync_dir", type=Path, required=True)
    ap.add_argument("--grace", type=float, default=900, help="seconds to wait for remotes per round")
    ap.add_argument("--start_round", type=int, default=1)
    args = ap.parse_args()
    d = args.sync_dir
    d.mkdir(parents=True, exist_ok=True)
    r = args.start_round
    while True:
        local = d / f"w0_r{r}.pt"
        while not local.exists():
            # skip ahead if the local worker resumed at a later round
            later = sorted(int(p.stem.split("_r")[1]) for p in d.glob("w0_r*.pt"))
            if later and later[-1] > r:
                r = later[-1]
                local = d / f"w0_r{r}.pt"
                break
            time.sleep(5)
        remotes = json.loads((d / "remotes.json").read_text()) if (d / "remotes.json").exists() else []
        got = [local]
        t0 = time.time()
        pending = list(remotes)
        while pending and time.time() - t0 < args.grace:
            for rem in list(pending):
                dst = d / f"w{rem['rank']}_r{r}.pt"
                if scp(f"{rem['host']}:{rem['dir']}/w{rem['rank']}_r{r}.pt", str(dst), rem.get("ssh", [])):
                    got.append(dst)
                    pending.remove(rem)
            if pending:
                time.sleep(20)
        states = [torch.load(p, map_location="cpu") for p in got]
        avg = {}
        for k in states[0]["w"]:
            avg[k] = sum(s["w"][k].float() for s in states).div_(len(states)).half()
        tmp = d / f".avg_r{r}.pt"
        torch.save({"w": avg, "n": len(states)}, tmp)
        tmp.rename(d / f"avg_r{r}.pt")
        for rem in remotes:
            if rem not in pending:
                scp(str(d / f"avg_r{r}.pt"), f"{rem['host']}:{rem['dir']}/avg_r{r}.pt", rem.get("ssh", []))
        print(f"{time.strftime('%F %T')} round {r}: averaged {len(states)} workers", flush=True)
        for old in d.glob("*_r*.pt"):
            try:
                if int(old.stem.split("_r")[1]) < r - 1:
                    old.unlink()
            except ValueError:
                pass
        r += 1


if __name__ == "__main__":
    main()
