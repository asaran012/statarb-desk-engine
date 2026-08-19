"""Cost sensitivity sweep, chunked so each invocation stays inside memory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from statarb.config import load_config
from statarb.data.loader import load_panel
from statarb.research.ablations import cost_sensitivity

OUT = Path("reports/cost_sensitivity.csv")


def main(bps_list):
    cfg = load_config("config/default.yaml")
    panel = load_panel(Path(cfg.data.processed_dir) / "panel.parquet")
    df = cost_sensitivity(panel, cfg, bps_grid=tuple(float(b) for b in bps_list))
    if OUT.exists():
        prev = pd.read_csv(OUT)
        df = pd.concat([prev[~prev["total_bps"].isin(df["total_bps"])], df],
                       ignore_index=True).sort_values("total_bps")
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"WROTE {len(df)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
