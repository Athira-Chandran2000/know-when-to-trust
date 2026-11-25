import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from kwtt.data import load_cases
from kwtt.metrics import dice

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--cache", required=True)
ap.add_argument("--table", default="outputs/dice_table.csv")
ap.add_argument("--out", default="outputs/stability_subsets.csv")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
VARIANTS = ["orig", "p1", "p2", "p3"]
rows = []
for c in load_cases(cfg["data_root"]):
    z = np.load(Path(args.cache) / f"{c['id']}.npz")
    for t in cfg["targets"]:
        m = {v: z[f"{t}__{v}"].astype(bool) for v in VARIANTS}
        pair = {(a, b): dice(m[a], m[b]) for a, b in combinations(VARIANTS, 2)}
        row = dict(case=c["id"], target=t)
        for k in (2, 3, 4):
            for extra in combinations(VARIANTS[1:], k - 1):
                S = ("orig",) + extra          # the original prompt is always asked
                row["S_" + "+".join(S)] = float(np.mean([pair[p] for p in combinations(S, 2)]))
        rows.append(row)

sub = pd.DataFrame(rows)
tab = pd.read_csv(args.table, dtype={"case": str})
orig = tab[tab.key == "orig"][["case", "target", "dice"]]
d = sub.merge(orig, on=["case", "target"])
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
d.to_csv(args.out, index=False)


def gain80(score, y):
    order = np.argsort(-score, kind="stable")
    return y[order[:int(round(0.8 * len(y)))]].mean() - y.mean()


res = []
for t in cfg["targets"]:
    g = d[d.target == t]
    y = g.dice.values
    for k in (2, 3, 4):
        cols = [c for c in g.columns if c.startswith("S_") and c.count("+") == k - 1]
        res.append(dict(target=t, variants=k,
                        spearman=round(float(np.mean([spearmanr(g[c].values, y)[0] for c in cols])), 2),
                        gain_at_80=round(float(np.mean([gain80(g[c].values, y) for c in cols])), 3)))
print("Stability with k prompt variants (averaged over the possible choices of extra variants):\n")
print(pd.DataFrame(res).to_string(index=False))