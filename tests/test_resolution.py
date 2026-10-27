# Tests for Phase 1 — ingestion, entity resolution, netting, reconciliation

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from config.loader import load_config
from cov.storage import Storage
from resolution import entity, ingest, netting, resolve_and_net_positions


@pytest.fixture
def cfg_tmp(tmp_path):
    cfg = load_config()
    new_paths = {}
    for key, value in cfg.paths.to_dict().items():
        tail = value.split("/data/", 1)[-1] if "/data/" in value else key
        new_paths[key] = str(tmp_path / "data" / tail) if key != "data_dir" else str(tmp_path / "data")
    cfg.paths._data = new_paths
    for k, v in new_paths.items():
        setattr(cfg.paths, k, v)
    cfg.ingestion.input_file = "tests/fixtures/positions_test.csv"
    cfg.ingestion.overrides_file = "tests/fixtures/entity_overrides_test.yaml"
    return cfg


@pytest.fixture
def book(cfg_tmp):
    from cov.pipeline import RunContext

    storage = Storage(cfg_tmp)
    ctx = RunContext(run_id="test", started_at=datetime(2026, 7, 20), storage=storage)
    status = resolve_and_net_positions(cfg_tmp, ctx)
    assert status.startswith("ok"), status
    yield ctx.artifacts["book"], ctx
    storage.close()


def _row(book, cid):
    hits = book[book["canonical_id"] == cid]
    assert len(hits) == 1, f"expected exactly one {cid!r}, got {len(hits)}"
    return hits.iloc[0]


def test_split_exchange_colon_prefix():
    prefixes = {"ETR", "ASX", "HKG"}
    assert ingest._split_exchange("etr:IFX", prefixes) == ("IFX", "ETR")
    assert ingest._split_exchange("ASX:LYC", prefixes) == ("LYC", "ASX")
    assert ingest._split_exchange("HKG:1913", prefixes) == ("1913", "HKG")
    assert ingest._split_exchange("CVE", prefixes) == ("CVE", None)


def test_parse_fee():
    assert ingest._parse_fee("0.53%") == 0.53
    assert ingest._parse_fee("0.03") == 0.03
    assert ingest._parse_fee("") is None
    assert ingest._parse_fee(None) is None


def test_reconciliation_invariant_to_the_cent(book):
    b, ctx = book
    assert int(b["market_value_usd"].sum()) == 4_632_000
    assert int(b["cost_basis_usd"].sum()) == 2_739_000
    assert ctx.artifacts["reconciliation"]["invariant_ok"] is True


def test_weights_sum_to_one(book):
    b, _ = book
    assert b["weight"].sum() == pytest.approx(1.0, abs=1e-9)


def test_asml_dual_isin_merges_and_sums_shares(book):
    b, _ = book
    asml = _row(b, "ASML")
    assert asml["quantity"] == 235
    assert asml["contributing_raw_ids"] == "19,20"
    assert "NL0010273215" in asml["isins"] and "USN070592100" in asml["isins"]


def test_ibit_cbtc_net_to_one_bitcoin_shares_not_additive(book):
    b, _ = book
    btc = _row(b, "BTC")
    ids = set(btc["contributing_raw_ids"].split(","))
    assert {"218", "220"} <= ids
    assert pd.isna(btc["quantity"])
    assert btc["is_fund"]
    assert "US46438F1012" in b[b["canonical_id"] == "BTC"].iloc[0]["isins"]


def test_boeing_and_bae_stay_separate(book):
    b, _ = book
    boeing = _row(b, "BA")
    bae = _row(b, "BAE")
    assert boeing["isins"] == "US0970231058"
    assert bae["isins"] == "GB0002634946"
    assert bae["quantity"] == 3035 + 1821


def test_bmw_share_classes_stay_separate(book):
    b, _ = book
    assert _row(b, "BMW")["isins"] == "DE0005190003"
    assert _row(b, "BMW3")["isins"] == "DE0005190037"


def test_googl_nets_across_custodians(book):
    b, _ = book
    googl = _row(b, "GOOGL")
    assert googl["quantity"] == 8000
    assert googl["contributing_raw_ids"] == "1,2"


def test_coin_is_equity_not_crypto(book):
    b, _ = book
    coin = _row(b, "COIN")
    assert coin["is_fund"] is False or coin["is_fund"] == False  # noqa: E712
    assert coin["instrument_type"] == "equity"


def test_fund_flagging(book):
    b, _ = book
    assert _row(b, "CSPX")["is_fund"]
    assert _row(b, "GDX")["is_fund"]


def test_data_quality_notes_surfaced(book):
    b, _ = book
    assert "price" in _row(b, "INDA")["resolution_notes"].lower()
    assert "0 market" in _row(b, "BMW3")["resolution_notes"].lower()


def test_snapshot_and_reference_tables_written(book):
    b, ctx = book
    assert "run=2026-07-20" in str(ctx.artifacts["book_path"])
    emap = ctx.storage.get_entity_map()
    assert {"ASML", "BTC", "BA", "BAE", "GOOGL"} <= set(emap["instrument_id"])
    meta = ctx.storage.get_instrument_metadata()
    assert (meta["currency"] == "USD").all()


def test_collision_guard_fires_on_undeclared_ticker_collision():
    df = pd.DataFrame({
        "raw_id": [1, 2],
        "custodian": ["UBS", "UBS"],
        "sector_label": ["x", "x"],
        "norm_ticker": ["BA", "BA"],
        "isin": ["US0970231058", "GB0002634946"],
        "name": ["Boeing", "BAE"],
        "shares": [1, 1],
        "cost_basis_usd": [1, 1],
        "market_value_usd": [1, 1],
        "etf_fee_pct": [None, None],
        "exchange": [None, None],
    })
    with pytest.raises(entity.ResolutionError):
        entity.resolve(df, {"merges": {}, "renames": [], "keep_separate": []})
