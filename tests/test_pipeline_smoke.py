# Smoke test: the whole pipeline runs end-to-end and writes a log

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import engine.fetch as fetch_mod
from config.loader import load_config
from cov.pipeline import PHASES, run_pipeline


class _OfflineFetcher(fetch_mod.PriceFetcher):
    def fetch_one(self, symbol, start, end):
        dates = pd.bdate_range(start, end)[:60]
        if not len(dates):
            return pd.DataFrame(columns=["date", "close"]), None
        if symbol.endswith("=X"):
            return pd.DataFrame({"date": dates, "close": [1.0] * len(dates)}), "USD"
        prices = 100 + np.arange(len(dates), dtype=float)
        return pd.DataFrame({"date": dates, "close": prices}), "USD"


@pytest.fixture
def offline_cfg(tmp_path, monkeypatch):
    cfg = load_config()
    new_paths = {}
    for key, value in cfg.paths.to_dict().items():
        tail = value.split("/data/", 1)[-1] if "/data/" in value else key
        new_paths[key] = str(tmp_path / "data" / tail) if key != "data_dir" else str(tmp_path / "data")
    new_paths["logs_dir"] = str(tmp_path / "logs")
    cfg.paths._data = new_paths
    for k, v in new_paths.items():
        setattr(cfg.paths, k, v)
    monkeypatch.setattr(fetch_mod, "YFinanceFetcher", _OfflineFetcher)
    return cfg


def test_pipeline_runs_end_to_end(offline_cfg):
    ctx = run_pipeline(offline_cfg)
    results = ctx.artifacts["results"]
    assert len(results) == len(PHASES)
    numbers = [r[0] for r in results]
    assert numbers == sorted(numbers)
    assert all(status != "error" for _, _, status in results)
    assert len(ctx.artifacts["book"]) > 100
    assert not ctx.artifacts["returns"].empty


def test_run_writes_a_log_file(offline_cfg):
    ctx = run_pipeline(offline_cfg)
    log_file = Path(offline_cfg.paths.logs_dir) / f"run_{ctx.run_id}.log"
    assert log_file.exists(), f"expected log file at {log_file}"
    assert log_file.stat().st_size > 0


def test_all_phases_present():
    numbers = [n for n, _, _ in PHASES]
    assert numbers == list(range(1, 9))
