"""Tests for Phase 2 — symbol resolution, FX-to-USD, log returns, quality gates.

Network is NOT hit: a FakePriceFetcher returns deterministic synthetic series, so
these tests are fast and offline. Symbol-mapping rules are tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import fetch, quality, returns as ret, symbols as sym


# --- symbol resolution -------------------------------------------------------
def _book(rows):
    return pd.DataFrame(rows, columns=["canonical_id", "ticker", "is_fund"])


def _meta(rows):
    return pd.DataFrame(rows, columns=["instrument_id", "listing_country"])


def test_auto_suffix_from_locale():
    book = _book([("IFX", "IFX", False), ("BAE", "BA", False), ("GOOGL", "GOOGL", False)])
    meta = _meta([("IFX", "DE"), ("BAE", "GB"), ("GOOGL", "US")])
    out = sym.resolve_symbols(book, meta, {}, {}).set_index("canonical_id")
    assert out.loc["IFX", "yahoo_symbol"] == "IFX.DE"
    assert out.loc["BAE", "yahoo_symbol"] == "BA.L"
    assert out.loc["GOOGL", "yahoo_symbol"] == "GOOGL"     # US -> no suffix


def test_dotted_ticker_used_verbatim():
    book = _book([("ENEL.MI", "ENEL.MI", False)])
    meta = _meta([("ENEL.MI", "IT")])          # would else become ENEL.MI.MI
    out = sym.resolve_symbols(book, meta, {}, {}).set_index("canonical_id")
    assert out.loc["ENEL.MI", "yahoo_symbol"] == "ENEL.MI"


def test_override_and_unmapped_and_crypto():
    book = _book([("BTC", "CBTC", True), ("ESLT", "ESLT", False), ("IUSVUSYS", "IUSVUSYS", True)])
    meta = _meta([("ESLT", "IL")])
    overrides = {"ESLT": {"symbol": "ESLT"}, "IUSVUSYS": {"symbol": None}}
    out = sym.resolve_symbols(book, meta, overrides, {"BTC": "BTC-USD"}).set_index("canonical_id")
    assert out.loc["BTC", "yahoo_symbol"] == "BTC-USD" and out.loc["BTC", "kind"] == "crypto_spot"
    assert out.loc["ESLT", "yahoo_symbol"] == "ESLT" and out.loc["ESLT", "kind"] == "override"
    assert out.loc["IUSVUSYS", "kind"] == "unmapped" and pd.isna(out.loc["IUSVUSYS", "yahoo_symbol"])


# --- FX / returns with a fake fetcher ----------------------------------------
class FakePriceFetcher(fetch.PriceFetcher):
    """Deterministic series: prices ramp 100,101,...; FX pairs are flat constants."""

    FX = {"EURUSD=X": 1.10, "GBPUSD=X": 1.25, "USDILS=X": 3.5}

    def fetch_one(self, symbol, start, end):
        dates = pd.bdate_range("2024-01-01", periods=10)
        if symbol in self.FX:
            return pd.DataFrame({"date": dates, "close": [self.FX[symbol]] * 10}), "USD"
        if symbol == "USDILS=X":
            return pd.DataFrame({"date": dates, "close": [3.5] * 10}), "ILS"
        if symbol == "MISSING":
            return pd.DataFrame(columns=["date", "close"]), None
        ccy = {"IFX.DE": "EUR", "BA.L": "GBp", "GOOGL": "USD"}.get(symbol, "USD")
        return pd.DataFrame({"date": dates, "close": np.arange(100, 110.0)}), ccy


def _symtab(rows):
    return pd.DataFrame(rows, columns=["canonical_id", "ticker", "yahoo_symbol", "kind"])


def test_fetch_skips_unmapped_and_records_failures():
    tab = _symtab([("A", "A", "GOOGL", "auto"), ("B", "B", None, "unmapped"),
                   ("C", "C", "MISSING", "auto")])
    prices, failures = fetch.fetch_prices(tab, "2024-01-01", "2024-01-31", FakePriceFetcher(), 2)
    assert set(prices["canonical_id"]) == {"A"}
    reasons = {f["canonical_id"]: f["reason"] for f in failures}
    assert reasons == {"B": "unmapped", "C": "no_data"}


def test_pence_conversion_divides_by_100():
    tab = _symtab([("BAE", "BA", "BA.L", "auto")])          # GBp
    prices, _ = fetch.fetch_prices(tab, "2024-01-01", "2024-01-31", FakePriceFetcher(), 1)
    usd, missing = ret.to_usd(prices, "2024-01-01", "2024-01-31", FakePriceFetcher())
    # 100 pence / 100 * 1.25 GBPUSD = 1.25
    assert usd["close_usd"].iloc[0] == pytest.approx(100 / 100 * 1.25)
    assert not missing


def test_eur_conversion_multiplies():
    tab = _symtab([("IFX", "IFX", "IFX.DE", "auto")])
    prices, _ = fetch.fetch_prices(tab, "2024-01-01", "2024-01-31", FakePriceFetcher(), 1)
    usd, _ = ret.to_usd(prices, "2024-01-01", "2024-01-31", FakePriceFetcher())
    assert usd["close_usd"].iloc[0] == pytest.approx(100 * 1.10)   # no /100 for EUR


def test_usd_passthrough_and_log_returns():
    tab = _symtab([("G", "G", "GOOGL", "auto")])
    prices, _ = fetch.fetch_prices(tab, "2024-01-01", "2024-01-31", FakePriceFetcher(), 1)
    usd, _ = ret.to_usd(prices, "2024-01-01", "2024-01-31", FakePriceFetcher())
    assert (usd["close_usd"] == usd["close"]).all()
    rl = ret.log_returns(usd)
    assert rl["return"].iloc[0] == pytest.approx(np.log(101 / 100))
    mat = ret.returns_matrix(rl)
    assert mat.shape == (9, 1) and "G" in mat.columns          # 10 prices -> 9 returns


# --- quality gates -----------------------------------------------------------
def test_quality_flags_short_history_and_stale():
    dates = pd.bdate_range("2024-01-01", periods=30)
    px = pd.DataFrame({"date": dates, "canonical_id": "X",
                       "close_usd": [100.0] * 30})            # totally flat -> stale
    rl = ret.log_returns(px)  # all zero returns
    gates = type("G", (), {"max_missing_pct": 5.0, "stale_price_days": 5,
                           "outlier_sigma": 4.0, "outlier_rolling_window": 21})()
    rep = quality.quality_report(px, rl, "t", gates, min_history_days=750)
    row = rep.iloc[0]
    assert row["flagged"]
    assert "short_history" in row["reasons"] and "stale_run" in row["reasons"]
    assert row["missing_pct"] >= 0                            # never negative (clamped)
