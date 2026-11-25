import numpy as np
from itertools import combinations
from scipy import ndimage
from .metrics import dice


def components(mask):
    """(number of connected pieces, fraction of the mask in the largest piece)."""
    if mask.sum() == 0:
        return 0, 0.0
    lab, n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), float(sizes.max() / sizes.sum())


def stability(masks):
    """Mean pairwise Dice between the prompt variants (needs no ground truth)."""
    return float(np.mean([dice(a, b) for a, b in combinations(masks, 2)]))


def adc_contrast(adc, mask, gland):
    """Mean ADC inside the mask divided by mean ADC in the rest of the gland.
    Cancer usually has LOWER ADC, so a ratio well below 1 supports a real lesion.
    NaN if the mask is empty, the shapes differ, or there is no tissue left to compare."""
    if adc.shape != mask.shape or mask.sum() == 0:
        return np.nan
    rest = gland & ~mask
    if rest.sum() == 0:
        return np.nan
    return float(adc[mask].mean() / (adc[rest].mean() + 1e-8))


def case_features(z, targets, voxel_ml, adc=None):
    """z: a loaded cache file. Returns one dict of features per target."""
    gland = z["gland__orig"].astype(bool)
    central = z["central__orig"].astype(bool)
    pz = z["pz__orig"].astype(bool)
    rows = []
    for t in targets:
        keys = [k for k in z.files if k.startswith(t + "__")]
        masks = [z[k].astype(bool) for k in keys]
        m = z[f"{t}__orig"].astype(bool)
        n_comp, largest = components(m)
        row = dict(target=t, pred_ml=float(m.sum()) * voxel_ml,
                   stability=stability(masks),
                   n_nonempty=sum(int(x.any()) for x in masks),
                   n_comp=n_comp, largest_frac=largest)
        if t in ("central", "pz", "lesion"):
            row["in_gland"] = float((m & gland).sum() / max(m.sum(), 1))
        if t in ("central", "pz"):
            other = pz if t == "central" else central
            row["overlap_other"] = float((m & other).sum() / max(m.sum(), 1))
            row["union_vs_gland"] = dice(central | pz, gland)
        if t == "lesion":
            row["adc_ratio"] = adc_contrast(adc, m, gland) if adc is not None else np.nan
        rows.append(row)
    return rows