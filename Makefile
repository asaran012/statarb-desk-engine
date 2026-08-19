.PHONY: setup test cov data validate research backtest ablations monitor report failure-demo all clean

PY ?= python3

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests -q

cov:
	$(PY) -m pytest tests -q --cov=src/statarb --cov-report=term-missing

data:
	$(PY) -m statarb.cli synth --config config/default.yaml

download:
	$(PY) -m statarb.cli download --config config/default.yaml

validate:
	$(PY) -m statarb.cli validate --config config/default.yaml

research:
	$(PY) -m statarb.cli research --config config/default.yaml

backtest:
	$(PY) -m statarb.cli backtest --config config/default.yaml

ablations:
	$(PY) -m statarb.cli ablations --config config/default.yaml

monitor:
	$(PY) -m statarb.cli monitor --config config/default.yaml

failure-demo:
	$(PY) scripts/run_failure_demo.py

report:
	$(PY) -m statarb.cli report --config config/default.yaml

all: data validate research backtest ablations monitor failure-demo report test

clean:
	rm -rf data/processed/* reports/figures/* reports/trades/* .pytest_cache __pycache__
