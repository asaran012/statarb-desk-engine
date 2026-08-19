"""Run a subset of ablation experiments and append results to reports/ablations.csv.

Chunked deliberately: the full study is ~7 walk-forward runs and holding all of
them in one process risks the memory ceiling on small machines.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from statarb.config import load_config
from statarb.data.loader import load_panel
from statarb.research.ablations import EXPERIMENTS, run_ablations

OUT = Path("reports/ablations.csv")


def main(keys):
    cfg = load_config("config/default.yaml")
    panel = load_panel(Path(cfg.data.processed_dir) / "panel.parquet")
    todo = [e for e in EXPERIMENTS if e.key in keys]
    if not todo:
        raise SystemExit(f"no experiments matched {keys}")
    df = run_ablations(panel, cfg, experiments=todo)
    if OUT.exists():
        prev = pd.read_csv(OUT)
        df = pd.concat([prev[~prev["key"].isin(df["key"])], df], ignore_index=True)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"WROTE {len(df)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
