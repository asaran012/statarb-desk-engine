"""Run the ablation study and cost sensitivity sweep. Writes reports/*.csv."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from statarb.config import load_config
from statarb.data.synthetic import generate_market
from statarb.research.ablations import (breakeven_cost_bps, cost_sensitivity,
                                        run_ablations)

def main():
    cfg = load_config("config/default.yaml")
    panel, gt = generate_market(cfg.data.start_date, cfg.data.end_date,
                                cfg.data.n_symbols, cfg.seed)
    Path("reports").mkdir(exist_ok=True)

    t0 = time.time()
    ab = run_ablations(panel, cfg)
    ab.to_csv("reports/ablations.csv", index=False)
    print(f"ABLATIONS DONE {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    cs = cost_sensitivity(panel, cfg, bps_grid=(0.0, 2.0, 5.0, 10.0, 20.0))
    cs.to_csv("reports/cost_sensitivity.csv", index=False)
    print(f"COST SWEEP DONE {time.time()-t1:.1f}s breakeven={breakeven_cost_bps(cs)}",
          flush=True)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
