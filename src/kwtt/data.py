"""Load Prostate158 cases and ground-truth masks."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_cases(data_root):
    root = Path(data_root)
    df = pd.concat(
        [pd.read_csv(root / "train.csv"), pd.read_csv(root / "valid.csv")],
        ignore_index=True,
    )
    cases = []
    for _, r in df.iterrows():
        cid = f"{int(r['ID']):03d}"
        d = root / "train" / cid
        has_lesion = isinstance(r["t2_tumor_reader1"], str)
        case = {
            "id": cid,
            "t2": d / "t2.nii",
            "adc": d / "adc.nii",
            "anatomy": d / "t2_anatomy_reader1.nii",
            "tumor": d / "t2_tumor_reader1.nii" if has_lesion else None,
            "has_lesion": has_lesion,
        }
        assert case["t2"].exists() and case["anatomy"].exists(), cid
        cases.append(case)
    return cases


def read_volume(path):
    """Read a NIfTI file reoriented to RAS (same reader VoxTell uses)."""
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    arr, _ = NibabelIOWithReorient().read_images([str(path)])
    return arr[0]


def ground_truth(case, target):
    """Boolean ground-truth mask. Labels: 1 = central gland, 2 = peripheral zone."""
    a = read_volume(case["anatomy"])
    if target == "gland":
        return a > 0
    if target == "central":
        return a == 1
    if target == "pz":
        return a == 2
    if target == "lesion":
        if case["tumor"] is None:
            return np.zeros(a.shape, dtype=bool)   # no lesion in this case
        return read_volume(case["tumor"]) > 0
    raise ValueError(target)