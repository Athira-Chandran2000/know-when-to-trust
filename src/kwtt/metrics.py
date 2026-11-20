import numpy as np


def dice(pred, gt):
    """Dice score. Both empty = 1.0 (correctly empty). One empty = 0.0."""
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return float(2 * (pred & gt).sum() / s)