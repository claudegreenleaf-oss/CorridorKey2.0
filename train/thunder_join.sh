#!/bin/bash
# Runs on the home box: waits for the Thunder worker's setup, then joins it to
# the running local-SGD job at the home worker's current step, starting from
# the coordinator's latest average.
set -euo pipefail
H=${1:-tnr-0}
CK=~/corridorkey
until ssh "$H" "grep -q SETUP_DONE ~/setup.log"; do
    ssh "$H" "tail -1 ~/setup.log" | grep -qi "error\|fatal" && { echo "setup failed"; exit 1; }
    sleep 30
done
avg=$(ls -t $CK/sync/avg_r*.pt | head -1)
step=$(grep -o "^step [0-9]*" $CK/ft/v3/train.log | tail -1 | cut -d" " -f2)
ssh "$H" "mkdir -p ~/sync ~/ft"
scp -q "$avg" "$H:sync/init.pt"
echo "[{\"host\": \"$H\", \"dir\": \"/home/ubuntu/sync\", \"rank\": 1}]" > $CK/sync/remotes.json
ssh "$H" "cd ~/ck2 && git pull -q; . venv/bin/activate && OPENCV_IO_ENABLE_OPENEXR=1 nohup python train/finetune.py \
  --data ~/data --out ~/ft --steps 200000 --save_every 1000000 --workers 7 --sync_dir ~/sync --rank 1 \
  --sync_every 2000 --lr 2e-5 --start_step $step --init_avg ~/sync/init.pt > ~/train.log 2>&1 &"
echo "joined at step $step from $(basename $avg)"
