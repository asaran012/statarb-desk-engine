import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statarb.config import load_config              # noqa: E402
from statarb.data.loader import to_wide             # noqa: E402
from statarb.data.synthetic import generate_market  # noqa: E402
from statarb.research.pair_selection import select_pairs  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config(str(ROOT / "config" / "default.yaml"))


@pytest.fixture(scope="session")
def market():
    panel, gt = generate_market("2016-01-03", "2021-12-31", 30, 20260818)
    return panel, gt


@pytest.fixture(scope="session")
def selection(market, config):
    panel, _ = market
    train = panel[panel["date"] <= pd.Timestamp("2018-12-31")]
    return select_pairs(train, max_pairs=10,
                        universe_kwargs=config.universe.model_dump())


@pytest.fixture(scope="session")
def wide(market):
    panel, _ = market
    return to_wide(panel, "close"), to_wide(panel, "open")
