"""P3M-10k train: estimate clean foregrounds (Germer multilevel) -> fg/*.jpg + alpha/*.png."""
import glob, io, os
from multiprocessing import Pool
import numpy as np, cv2, pyarrow.parquet as pq
root = os.path.expanduser(os.environ.get("P3M_ROOT", "~/corridorkey/data/p3m"))
os.makedirs(f"{root}/fg", exist_ok=True); os.makedirs(f"{root}/alpha", exist_ok=True)
def work(job):
    shard, g = job
    pf = pq.ParquetFile(shard)
    rows = pf.read_row_group(g).to_pylist()
    base = sum(pf.metadata.row_group(k).num_rows for k in range(g))
    return sum(one(shard, base + j, r) for j, r in enumerate(rows))
def one(shard, i, row):
    name = f"{os.path.basename(shard)[:-8]}_{i:05d}"
    if os.path.exists(f"{root}/alpha/{name}.png"):
        return 1
    from pymatting import estimate_foreground_ml
    img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
    a = cv2.imdecode(np.frombuffer(row["mask"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None or a is None or img.shape[:2] != a.shape[:2]:
        return 0
    fg = estimate_foreground_ml(img[..., ::-1] / 255.0, a / 255.0)
    cv2.imwrite(f"{root}/fg/{name}.jpg", (np.clip(fg, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(f"{root}/alpha/{name}.png", a)
    return 1
jobs = [(s, g) for s in sorted(glob.glob(f"{root}/train-*.parquet")) for g in range(pq.ParquetFile(s).num_row_groups)]
print(len(jobs), "row groups", flush=True)
with Pool(min(24, os.cpu_count())) as p:
    print("P3M", sum(p.imap_unordered(work, jobs)), flush=True)
