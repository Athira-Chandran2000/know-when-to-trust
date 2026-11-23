import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import yaml
from kwtt.data import load_cases, ground_truth
from kwtt.metrics import dice

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--cache", required=True, help="folder with the cached <case>.npz files")
ap.add_argument("--out", default="outputs/dice_table.csv")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
rows = []
for c in load_cases(cfg["data_root"]):
    z = np.load(Path(args.cache) / f"{c['id']}.npz")
    zooms = nib.load(str(c["t2"])).header.get_zooms()[:3]
    voxel_ml = float(np.prod(zooms)) / 1000.0           # mm^3 -> ml
    for t in cfg["targets"]:
        gt = ground_truth(c, t)
        for k in [k for k in z.files if k.startswith(t + "__")]:
            pred = z[k].astype(bool)
            rows.append(dict(case=c["id"], target=t, key=k.split("__")[1],
                             dice=dice(pred, gt), pred_ml=pred.sum() * voxel_ml,
                             gt_ml=gt.sum() * voxel_ml, voxel_ml=voxel_ml,
                             has_lesion=c["has_lesion"]))

df = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, index=False)
print("saved", args.out, df.shape)

o = df[df.key == "orig"]
for t, spec in cfg["targets"].items():
    d = o[o.target == t]
    thr = spec["failure_dice"]
    if t == "lesion":
        pos, neg = d[d.has_lesion], d[~d.has_lesion]
        print(f"lesion, label exists (n={len(pos)}): mean Dice {pos.dice.mean():.3f}, missed (<{thr}): {(pos.dice < thr).mean():.0%}")
        print(f"lesion, no label   (n={len(neg)}): predicted a lesion anyway: {(neg.pred_ml > 0).mean():.0%}")
    else:
        print(f"{t:8s} n={len(d)}  mean Dice {d.dice.mean():.3f}  min {d.dice.min():.2f}  failures (<{thr}): {(d.dice < thr).mean():.0%}")