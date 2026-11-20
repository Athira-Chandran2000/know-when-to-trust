import argparse
import yaml
from kwtt.data import load_cases
from kwtt.tools.voxtell import VoxTellTool
from kwtt.cache import cache_voxtell

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--limit", type=int, default=None, help="only the first N cases (for testing)")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
cases = load_cases(cfg["data_root"])
if args.limit:
    cases = cases[:args.limit]
cache_voxtell(cases, cfg, VoxTellTool(), f"{cfg['cache_dir']}/voxtell")