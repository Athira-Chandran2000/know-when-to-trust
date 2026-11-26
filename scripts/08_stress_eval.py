import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from kwtt.data import load_cases, ground_truth
from kwtt.metrics import dice

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--clean", required=True, help="folder with the clean cache (voxtell)")
ap.add_argument("--stress", required=True, help="folder with the stress cache")
ap.add_argument("--features", default="outputs/features.csv")
ap.add_argument("--out_dir", default="outputs")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
VARS = ["orig", "p1", "p2", "p3"]
targets = list(cfg["targets"])
cases = {c["id"]: c for c in load_cases(cfg["data_root"])}
stress_ids = sorted(p.stem for p in Path(args.stress).glob("*.npz") if ".tmp" not in p.name)
print("stress cases:", len(stress_ids))


def stab(masks):
    return float(np.mean([dice(a, b) for a, b in combinations(masks, 2)]))


# threshold calibrated on clean cases that are NOT in the stress set (no ground truth used)
feat = pd.read_csv(args.features, dtype={"case": str})
rest = feat[~feat.case.isin(stress_ids)]
tau = {t: float(np.percentile(rest[rest.target == t].stability, 20)) for t in targets}
tau_g = float(np.percentile(rest[rest.target.isin(["central", "pz"])].stability, 20))

# ---------- A. noise and blur ----------
rows = []
for cid in stress_ids:
    zc = np.load(Path(args.clean) / f"{cid}.npz")
    zs = np.load(Path(args.stress) / f"{cid}.npz")
    for t in targets:
        gt = ground_truth(cases[cid], t)
        for cond, z, pre in [("clean", zc, ""), ("noise", zs, "noise__"), ("blur", zs, "blur__")]:
            masks = [z[f"{pre}{t}__{v}"].astype(bool) for v in VARS]
            rows.append(dict(case=cid, target=t, cond=cond, dice=dice(masks[0], gt), stability=stab(masks)))
res = pd.DataFrame(rows)
res["abstain"] = [s < tau[t] for s, t in zip(res.stability, res.target)]
res.to_csv(Path(args.out_dir) / "stress_degradation.csv", index=False)

print("\n=== A. Image degradation (30 cases) ===")
for t in targets:
    g = res[res.target == t]
    a = g.groupby("cond").agg(mean_dice=("dice", "mean"), mean_stability=("stability", "mean"),
                              abstain_rate=("abstain", "mean"))
    kept = g[~g.abstain].groupby("cond").dice.mean().rename("dice_of_accepted")
    print(f"\n{t}: abstain if stability < {tau[t]:.3f}")
    print(a.join(kept).round(3).to_string())
    print("   pooled Spearman(stability, Dice):", round(float(spearmanr(g.stability, g.dice)[0]), 2))

# ---------- B. wrong-organ prompts ----------
wr = []
for cid in stress_ids:
    zc = np.load(Path(args.clean) / f"{cid}.npz")
    zs = np.load(Path(args.stress) / f"{cid}.npz")
    gland = zc["gland__orig"].astype(bool)
    items = [("wrong", n, zs, f"wrong__{n}__") for n in cfg["stress"]["wrong"]]
    items += [("valid", t, zc, f"{t}__") for t in ("central", "pz")]
    for kind, name, z, pre in items:
        masks = [z[f"{pre}{v}"].astype(bool) for v in VARS]
        m = masks[0]
        wr.append(dict(case=cid, kind=kind, request=name, stability=stab(masks),
                       pred_empty=int(not m.any()),
                       in_gland=float((m & gland).sum() / max(m.sum(), 1))))
w = pd.DataFrame(wr)
w["abstain_stability"] = (w.pred_empty == 1) | (w.stability < tau_g)
w["abstain_in_gland"] = (w.pred_empty == 1) | (w.in_gland < 0.5)
w["abstain_either"] = w.abstain_stability | w.abstain_in_gland
w.to_csv(Path(args.out_dir) / "stress_wrong_organ.csv", index=False)

print("\n=== B. Wrong-organ requests (should abstain) vs valid requests (should not) ===")
print(f"stability threshold: {tau_g:.3f}; in-gland threshold: 0.5")
print(w.groupby(["kind", "request"])[["pred_empty", "abstain_stability", "abstain_in_gland", "abstain_either"]]
      .mean().round(2).to_string())