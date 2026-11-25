import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--features", default="outputs/features.csv")
ap.add_argument("--out_dir", default="outputs")
ap.add_argument("--repeats", type=int, default=5)
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
df = pd.read_csv(args.features, dtype={"case": str})
FEATS = ["stability", "n_nonempty", "pred_ml", "largest_frac", "n_comp",
         "in_gland", "overlap_other", "union_vs_gland", "adc_ratio"]


def oof_predictions(X, y, repeats, seed=0):
    """Out-of-fold predicted Dice: every case is predicted by models that never saw it."""
    preds = np.zeros((repeats, len(y)))
    for r in range(repeats):
        for tr, te in KFold(5, shuffle=True, random_state=seed + r).split(X):
            m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                      random_state=seed + r, n_jobs=-1)
            m.fit(X[tr], y[tr])
            preds[r, te] = m.predict(X[te])
    return preds.mean(axis=0)


def selective_dice(score, dice, coverages):
    """Mean true Dice of the cases kept when the agent keeps the top-scoring fraction."""
    order = np.argsort(-score, kind="stable")
    return {c: float(dice[order[:max(1, int(round(c * len(dice))))]].mean()) for c in coverages}


out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
grid = [float(g) for g in np.linspace(0.3, 1.0, 15)]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
summary, oof_rows = [], []

for ax, (t, spec) in zip(axes, cfg["targets"].items()):
    d = df[df.target == t].reset_index(drop=True)
    feats = [f for f in FEATS if f in d and d[f].notna().any()]
    X = d[feats].copy()
    if "adc_ratio" in X:
        X["adc_ratio"] = X["adc_ratio"].fillna(1.0)
    X = X.fillna(0).values
    y = d["dice"].values
    pred = oof_predictions(X, y, args.repeats)
    stab = d["stability"].values

    if t == "lesion":      # fixed in advance: Dice < 0.1 counts as a failure (absent lesions have Dice 0)
        fail, fdef = y < spec["failure_dice"], f"Dice<{spec['failure_dice']}"
    else:                  # decided after seeing the data: worst quartile
        fail, fdef = y <= np.quantile(y, 0.25), "worst quartile (post hoc)"

    at = lambda sc, c: selective_dice(sc, y, [c])[c]
    summary.append(dict(
        target=t, n=len(y), accept_all=round(float(y.mean()), 3),
        agent_80=round(at(pred, 0.8), 3), stability_80=round(at(stab, 0.8), 3), oracle_80=round(at(y, 0.8), 3),
        agent_50=round(at(pred, 0.5), 3), stability_50=round(at(stab, 0.5), 3), oracle_50=round(at(y, 0.5), 3),
        spearman=round(float(spearmanr(pred, y)[0]), 2),
        auroc_agent=round(float(roc_auc_score(fail, -pred)), 2),
        auroc_stability=round(float(roc_auc_score(fail, -stab)), 2),
        failure_def=fdef))

    for name, sc in [("agent (cross-validated)", pred), ("stability only", stab), ("oracle", y)]:
        cv = selective_dice(sc, y, grid)
        ax.plot(grid, [cv[g] for g in grid], label=name)
    ax.axhline(float(y.mean()), ls="--", color="gray", label="random abstention")
    ax.set_title(t); ax.set_xlabel("coverage (fraction of cases kept)"); ax.set_ylabel("mean Dice of kept cases")
    oof_rows.append(d.assign(pred_dice=pred)[["case", "target", "dice", "pred_dice"]])

axes[0].legend(fontsize=8)
plt.tight_layout()
fig.savefig(out_dir / "risk_coverage.png", dpi=150)
pd.concat(oof_rows).to_csv(out_dir / "oof_predictions.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))
print("\nsaved", out_dir / "risk_coverage.png")