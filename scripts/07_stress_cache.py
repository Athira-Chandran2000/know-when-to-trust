import argparse
from pathlib import Path
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from kwtt.data import load_cases, read_volume
from kwtt.tools.voxtell import VoxTellTool
from kwtt.cache import prompt_table

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--n_cases", type=int, default=30)
ap.add_argument("--limit", type=int, default=None, help="only the first N chosen cases (testing)")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
out_dir = Path(cfg["cache_dir"]) / "stress"
out_dir.mkdir(parents=True, exist_ok=True)

cases = load_cases(cfg["data_root"])
rng = np.random.default_rng(cfg["seed"])
idx = sorted(int(i) for i in rng.choice(len(cases), size=args.n_cases, replace=False))
cases = [cases[i] for i in idx]
if args.limit:
    cases = cases[:args.limit]

tool = VoxTellTool()
bs = cfg.get("prompt_batch", 4)
right = prompt_table(cfg)                      # the normal prompts: 4 targets x 4 variants
wrong = []
for name, spec in cfg["stress"]["wrong"].items():
    wrong.append((f"{name}__orig", spec["prompt"]))
    for i, p in enumerate(spec["paraphrases"], 1):
        wrong.append((f"{name}__p{i}", p))


def run(img, items):
    keys = [k for k, _ in items]
    prompts = [p for _, p in items]
    masks = []
    for i in range(0, len(prompts), bs):
        masks += tool.predict(img, prompts[i:i + bs])
    return {k: m.astype(np.uint8) for k, m in zip(keys, masks)}


def perturb(img, kind, seed):
    x = img.astype(np.float32)                 # shape (1, z, y, x)
    if kind == "noise":
        r = np.random.default_rng(seed)
        return x + r.normal(0, 0.15 * x.std(), x.shape).astype(np.float32)
    return gaussian_filter(x, sigma=(0, 0, 3, 3))      # blur in-plane only


print("stress cases:", [c["id"] for c in cases])
for c in tqdm(cases):
    final = out_dir / f"{c['id']}.npz"
    if final.exists():
        continue
    img = read_volume(c["t2"])[None]
    out = {}
    for cond in ("noise", "blur"):
        for k, m in run(perturb(img, cond, int(c["id"])), right).items():
            out[f"{cond}__{k}"] = m
    for k, m in run(img, wrong).items():
        out[f"wrong__{k}"] = m
    tmp = out_dir / f"{c['id']}.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.rename(final)