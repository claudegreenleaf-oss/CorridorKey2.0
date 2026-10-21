#!/usr/bin/env python3
"""Deletes the Thunder worker as soon as it is not needed (runs on the home box).

Deletes the instance when any of these holds:
  * estimated spend since it started running reaches --budget (default $47,
    leaving margin under the hard $50 cap);
  * the home training job (systemd unit --train_unit) is no longer running;
  * the remote worker has not produced a sync file for --stall minutes;
  * the file --stop_file exists (manual stop).
Spend is hours RUNNING times --rate, checked every minute.
"""
import argparse, json, subprocess, time
from pathlib import Path


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="0")
    ap.add_argument("--rate", type=float, default=0.35 + 0.06)   # GPU + vCPU/storage margin
    ap.add_argument("--budget", type=float, default=47.0)
    ap.add_argument("--train_unit", default="ck-train")
    ap.add_argument("--sync_dir", type=Path, default=Path.home() / "corridorkey/sync")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--stall", type=float, default=150)
    ap.add_argument("--stop_file", type=Path, default=Path.home() / "corridorkey/STOP_THUNDER")
    ap.add_argument("--state", type=Path, default=Path.home() / "corridorkey/thunder_watchdog.json")
    a = ap.parse_args()
    st = json.loads(a.state.read_text()) if a.state.exists() else {"running_s": 0.0, "last_seen": None}
    last = time.time()
    while True:
        time.sleep(60)
        now = time.time()
        status = sh("tnr status --no-wait 2>&1")
        running = any(l.split()[:1] == [a.instance] and "RUNNING" in l for l in status.splitlines())
        if running:
            st["running_s"] += now - last
        last = now
        spend = st["running_s"] / 3600 * a.rate
        files = sorted(a.sync_dir.glob(f"w{a.rank}_r*.pt"), key=lambda p: p.stat().st_mtime)
        if files:
            st["last_seen"] = files[-1].stat().st_mtime
        reason = None
        if spend >= a.budget:
            reason = f"budget reached (${spend:.2f})"
        elif sh(f"systemctl --user is-active {a.train_unit}").strip() != "active":
            reason = "home training job is not running"
        elif a.stop_file.exists():
            reason = "manual stop"
        elif running and st["last_seen"] and now - st["last_seen"] > a.stall * 60:
            reason = f"worker stalled ({(now - st['last_seen']) / 60:.0f} min without a sync file)"
        st["spend"] = round(spend, 2)
        a.state.write_text(json.dumps(st))
        if reason:
            print(time.strftime("%F %T"), "deleting instance:", reason, flush=True)
            print(sh(f"tnr delete {a.instance} -y 2>&1")[-500:], flush=True)
            return
        print(time.strftime("%F %T"), f"running={running} spend=${spend:.2f}", flush=True)


if __name__ == "__main__":
    main()
