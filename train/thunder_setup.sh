#!/bin/bash
# Unattended setup of a CorridorKey 3.0 training worker on a fresh cloud GPU box.
# Pulls code, weights and the public train splits straight from GitHub / Hugging Face
# (no upload from home). Renders and AM-2k stay on the home box.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg unzip git python3-venv > /dev/null
cd ~
[ -d ck2 ] || git clone -q -b fidelity https://github.com/claudegreenleaf-oss/CorridorKey2.0 ck2
cd ck2 && git pull -q
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt pyarrow pymatting scipy
python scripts/download_weights.py > /dev/null
mkdir -p ~/data && cd ~/data
H=https://huggingface.co/datasets
if [ ! -d VideoMatte240K_JPEG_HD/train ]; then
  wget -q $H/geyongtao/video_matting/resolve/main/videomatte.zip -O vm.zip
  unzip -q vm.zip "VideoMatte240K_JPEG_HD/train/*" && rm vm.zip
fi
mkdir -p t460/Train/fg t460/Train/alpha
curl -s "https://huggingface.co/api/datasets/Thinnaphat/transparent-460/tree/main?recursive=true" | python3 -c "
import sys,json
for x in json.load(sys.stdin):
    p=x['path']
    if x['type']=='file' and (p.startswith('Train/fg/') or p.startswith('Train/alpha/')): print(p)" \
  | xargs -P 8 -I{} wget -q -nc "$H/Thinnaphat/transparent-460/resolve/main/{}" -O "t460/{}" || true
mkdir -p p3m
for i in $(seq -w 0 10); do echo "$H/nobg/P3M-10K/resolve/main/data/train-000${i}-of-00011.parquet"; done | xargs -P 6 -I{} wget -q -nc {} -P p3m/
cd ~/ck2 && P3M_ROOT=~/data/p3m OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python train/prep_p3m.py
echo SETUP_DONE
