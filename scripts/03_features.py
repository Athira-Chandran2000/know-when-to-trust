import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from kwtt.data import load_cases
from kwtt.verifier import case_features

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--cache", required=True)
ap.add_argument("--table", default="outputs/dice_table.csv")
ap.add_argument("--out", default="outputs/features.csv")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
rows = []
for c in load_cases(cfg["data_root"]):
    z = np.load(Path(args.cache) / f"{c['id']}.npz")
    zooms = nib.load(str(c["t2"])).header.get_zooms()[:3]
    voxel_ml = float(np.prod(zooms)) / 1000.0
    for r in case_features(z, list(cfg["targets"]), voxel_ml):
        r["case"] = c["id"]
        rows.append(r)
feat = pd.DataFrame(rows)

tab = pd.read_csv(args.table, dtype={"case": str})
orig = tab[tab.key == "orig"][["case", "target", "dice", "gt_ml", "has_lesion"]]
out = feat.merge(orig, on=["case", "target"])
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out, index=False)
print("saved", args.out, out.shape)

cols = ["stability", "pred_ml", "largest_frac", "in_gland", "overlap_other", "union_vs_gland"]
print("\nSpearman correlation of each check with the true Dice:")
for t in cfg["targets"]:
    d = out[out.target == t]
    if t == "lesion":
        d = d[d.has_lesion]
    res = {c: round(float(spearmanr(d[c], d.dice, nan_policy="omit")[0]), 2)
           for c in cols if c in d and d[c].notna().any()}
    print(f"  {t:8s} n={len(d)}", res)

l = out[out.target == "lesion"]
print("\nLesion: mean of each check, lesion absent (False) vs present (True):")
print(l.groupby("has_lesion")[["stability", "n_nonempty", "pred_ml", "in_gland"]].mean().round(2))
print("\nLesion absent vs present, single-check AUROC (direction-free, descriptive only):")
for c in ["stability", "n_nonempty", "pred_ml", "in_gland"]:
    a = roc_auc_score(~l.has_lesion, l[c])
    print(f"  {c:11s} {max(a, 1 - a):.2f}")