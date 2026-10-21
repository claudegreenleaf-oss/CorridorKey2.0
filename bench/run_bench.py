#!/usr/bin/env python3
"""Run CorridorKey over every benchmark case, then score it.

  python bench/run_bench.py --bench ~/corridorkey/bench --out ~/corridorkey/results/baseline
  python bench/run_bench.py ... --name fixed -- --decode ffmpeg --precision mixed

Anything after `--` is passed straight to infer.py, so one script covers the
upstream baseline and every fix variant.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--repo", type=Path, default=REPO,
                    help="CorridorKey checkout whose infer.py to run")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    timings = {}
    for case_dir in sorted(p for p in args.bench.iterdir() if (p / "case.json").exists()):
        if args.only and case_dir.name not in args.only:
            continue
        meta = json.loads((case_dir / "case.json").read_text())
        out = args.out / case_dir.name
        if args.skip_existing and list(out.glob("*_alpha_exr")):
            continue
        uhd = meta["height"] > 1500
        cmd = [sys.executable, str(args.repo / "infer.py"),
               "--input", str(case_dir / meta["plate"]),
               "--output_dir", str(out),
               "--output_format", "exr", "--outputs", "alpha", "fg",
               "--hann_chunk", "16" if uhd else "32",
               "--hann_stride", "8" if uhd else "16",
               "--frame_dir_fps", "24"]
        if meta.get("linear_input"):
            cmd.append("--frame_dir_linear")
        cmd += extra
        t0 = time.time()
        print(f"== {case_dir.name}", flush=True)
        r = subprocess.run(cmd, cwd=args.repo, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        (out.parent / f"{case_dir.name}.log").write_text(r.stdout)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit(f"{case_dir.name} failed")
        timings[case_dir.name] = round(time.time() - t0, 1)
        print(f"   {timings[case_dir.name]} s", flush=True)
    (args.out / "timings.json").write_text(json.dumps(timings, indent=1))
    subprocess.run([sys.executable, str(REPO / "bench" / "eval_bench.py"),
                    "--bench", str(args.bench), "--results", str(args.out)]
                   + (["--only", *args.only] if args.only else []), check=True)


if __name__ == "__main__":
    main()
