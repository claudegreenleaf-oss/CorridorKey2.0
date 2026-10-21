#!/usr/bin/env python3
"""Compare benchmark scores across variants.

  python bench/compare.py ~/corridorkey/results baseline decode mixed all
"""

import json
import sys
from pathlib import Path

from eval_bench import LOWER_IS_BETTER

SHOW = ["comp_mae", "comp_edge", "comp_psnr", "sad", "edge_mae", "holes", "leak", "dtssd", "fg_mae"]


def main():
    root = Path(sys.argv[1])
    variants = sys.argv[2:]
    scores = {v: json.loads((root / v / "scores.json").read_text()) for v in variants}
    base = variants[0]
    cases = sorted(set.intersection(*(set(s) for s in scores.values())))
    for m in SHOW:
        print(f"\n## {m} ({'lower' if LOWER_IS_BETTER[m] else 'higher'} is better)")
        print(f"{'case':26s}" + "".join(f"{v:>14s}" for v in variants))
        means = {v: 0.0 for v in variants}
        for c in cases:
            row = f"{c:26s}"
            for v in variants:
                x = scores[v][c][m]
                means[v] += x / len(cases)
                if v == base:
                    row += f"{x:14.3f}"
                else:
                    b = scores[base][c][m]
                    d = (x - b) / abs(b) * 100 if b else 0.0
                    row += f"{x:8.3f}{d:+5.0f}%"
            print(row)
        row = f"{'MEAN':26s}"
        for v in variants:
            b = means[base]
            row += f"{means[v]:14.3f}" if v == base else f"{means[v]:8.3f}{(means[v] - b) / abs(b) * 100:+5.0f}%"
        print(row)


if __name__ == "__main__":
    main()
