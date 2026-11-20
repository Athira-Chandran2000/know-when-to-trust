from pathlib import Path
import numpy as np
from tqdm import tqdm
from .data import read_volume


def prompt_table(cfg):
    """List of (key, prompt): the original prompt and paraphrases for every target."""
    items = []
    for target, spec in cfg["targets"].items():
        items.append((f"{target}__orig", spec["prompt"]))
        for i, p in enumerate(spec.get("paraphrases", []), 1):
            items.append((f"{target}__p{i}", p))
    return items


def cache_voxtell(cases, cfg, tool, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = prompt_table(cfg)
    keys = [k for k, _ in table]
    prompts = [p for _, p in table]
    bs = cfg.get("prompt_batch", 4)

    todo = [c for c in cases if not (out_dir / f"{c['id']}.npz").exists()]
    print(f"{len(cases) - len(todo)} cases already cached, {len(todo)} to do")

    for c in tqdm(todo):
        img = read_volume(c["t2"])[None]            # (1, z, y, x)
        masks = []
        for i in range(0, len(prompts), bs):
            masks += tool.predict(img, prompts[i:i + bs])
        tmp = out_dir / f"{c['id']}.tmp.npz"
        np.savez_compressed(tmp, **{k: m.astype(np.uint8) for k, m in zip(keys, masks)})
        tmp.rename(out_dir / f"{c['id']}.npz")    # only finished cases get the final name