# Tests for the storage layer (Parquet system-of-record + DuckDB query layer)

from __future__ import annotations

import copy

import pandas as pd
import pytest

from config.loader import load_config
from cov.storage import Storage


@pytest.fixture
def storage(tmp_path):
    cfg = load_config()
    new_paths = {}
    for key, value in cfg.paths.to_dict().items():
        tail = value.split("/data/", 1)[-1] if "/data/" in value else key
        new_paths[key] = str(tmp_path / "data" / tail) if key != "data_dir" else str(tmp_path / "data")
    cfg.paths._data = new_paths
    for k, v in new_paths.items():
        setattr(cfg.paths, k, v)
    s = Storage(cfg)
    yield s
    s.close()


def test_reference_tables_created_empty(storage):
    assert storage.get_entity_map().empty
    assert storage.get_instrument_metadata().empty


def test_upsert_and_read_entity_map(storage):
    df = pd.DataFrame(
        [
            {"instrument_id": "BTC", "canonical_name": "Bitcoin", "isin": None,
             "ticker": "BTC-USD", "instrument_type": "etp", "note": "IBIT+CBTC net here"},
            {"instrument_id": "ASML", "canonical_name": "ASML Holding", "isin": "NL0010273215",
             "ticker": "ASML", "instrument_type": "equity", "note": "dual listing"},
        ]
    )
    n = storage.upsert_reference("entity_map", df)
    assert n == 2
    out = storage.get_entity_map()
    assert set(out["instrument_id"]) == {"BTC", "ASML"}

    storage.upsert_reference("entity_map", df.head(1))
    assert len(storage.get_entity_map()) == 2


def test_upsert_rejects_unknown_table(storage):
    with pytest.raises(ValueError):
        storage.upsert_reference("not_a_table", pd.DataFrame())


def test_prices_roundtrip_append_only(storage):
    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "close": [185.0, 184.2, 370.0],
            "return": [0.0, -0.0043, 0.0],
        }
    )
    path = storage.write_prices(df, year=2024)
    assert path.exists()

    prices = storage.get_prices(tickers=["AAPL"])
    assert set(prices["ticker"]) == {"AAPL"}
    assert len(prices) == 2

    panel = storage.get_returns(start="2024-01-01", end="2024-12-31")
    assert "AAPL" in panel.columns and "MSFT" in panel.columns


def test_reads_are_empty_before_any_write(storage):
    assert storage.get_prices().empty
    assert storage.get_returns().empty
    assert storage.get_factors().empty
    assert storage.get_holdings().empty


def test_snapshot_write_is_partitioned_by_run(storage):
    book = pd.DataFrame({"instrument_id": ["AAPL"], "weight": [0.1]})
    path = storage.write_snapshot("book", "2026-07-18", book, name="netted_book")
    assert path.exists()
    assert "run=2026-07-18" in str(path)

    with pytest.raises(ValueError):
        storage.write_snapshot("bogus", "2026-07-18", book, name="x")
