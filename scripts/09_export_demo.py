import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="outputs/features.csv")
ap.add_argument("--oof", default="outputs/oof_predictions.csv")
ap.add_argument("--out", default="app/demo_data.csv")
args = ap.parse_args()

f = pd.read_csv(args.features, dtype={"case": str})[["case", "target", "stability"]]
o = pd.read_csv(args.oof, dtype={"case": str})
d = o.merge(f, on=["case", "target"])
d = d.sample(frac=1, random_state=0).drop(columns="case").reset_index(drop=True)   # no case IDs
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
d.round(4).to_csv(args.out, index=False)
print("saved", args.out, d.shape)