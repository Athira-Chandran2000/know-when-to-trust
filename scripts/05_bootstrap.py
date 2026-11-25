import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="outputs/features.csv")
ap.add_argument("--oof", default="outputs/oof_predictions.csv")
ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

f = pd.read_csv(args.features, dtype={"case": str})[["case", "target", "stability"]]
o = pd.read_csv(args.oof, dtype={"case": str})
d = o.merge(f, on=["case", "target"])


def kept_mean(score, y, cov):
    order = np.argsort(-score, kind="stable")
    return y[order[:max(1, int(round(cov * len(y))))]].mean()


def ci(x):
    return f"{np.mean(x):+.3f} [{np.percentile(x, 2.5):+.3f}, {np.percentile(x, 97.5):+.3f}]"


rng = np.random.default_rng(args.seed)
rows = []
for t, g in d.groupby("target", sort=False):
    y, p, s = g.dice.values, g.pred_dice.values, g.stability.values
    n = len(y)
    ga, gs, diff = [], [], []
    for _ in range(args.n_boot):
        i = rng.integers(0, n, n)
        a = kept_mean(p[i], y[i], 0.8) - y[i].mean()
        b = kept_mean(s[i], y[i], 0.8) - y[i].mean()
        ga.append(a); gs.append(b); diff.append(a - b)
    rows.append(dict(target=t, gain_agent_at_80=ci(ga), gain_stability_at_80=ci(gs),
                     agent_minus_stability=ci(diff)))
print("Gain in mean Dice from abstaining on the least-confident 20% (95% bootstrap CI):\n")
print(pd.DataFrame(rows).to_string(index=False))