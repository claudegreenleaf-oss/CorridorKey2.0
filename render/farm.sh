#!/bin/bash
# Keep rendering keying clips forever: N parallel Blender workers, unique seeds.
#   render/farm.sh <out_root> <workers> <threads_per_worker>
OUT=${1:-$HOME/corridorkey/data/renders}
N=${2:-2}
TH=${3:-32}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
worker() {
  local w=$1
  while true; do
    seed=$(( (RANDOM << 15 | RANDOM) ))
    dir="$OUT/clip_$(printf %010d $seed)"
    [ -e "$dir" ] && continue
    /opt/blender/blender -b --factory-startup -P "$REPO/render/gen_scene.py" -- \
      --seed $seed --out "$dir" --frames 8 --threads $TH > "$dir.log" 2>&1
    if [ -f "$dir/done" ]; then rm -f "$dir.log"; else rm -rf "$dir"; fi
  done
}
for w in $(seq 1 $N); do worker $w & done
wait
