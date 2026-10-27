# Tests for Phase 3 — ETF look-through (offline, fixture-driven)

from __future__ import annotations

import pandas as pd
import pytest

from resolution import holdings_parse, lookthrough


def _book() -> pd.DataFrame:
    rows = [
        ("AAPL", "AAPL", "US0378331005", "Apple Inc", 0.10, False),
        ("MSFT", "MSFT", "US5949181045", "Microsoft Corp", 0.05, False),
        ("SPY", "SPY", "US78462F1030", "SPDR S&P 500 ETF", 0.20, True),
        ("QQQ", "QQQ", "US46090E1038", "Invesco QQQ", 0.10, True),
        ("GLD", "GLD", "US78463V1070", "SPDR Gold Shares", 0.05, True),
        ("NFLX", "NFLX", "US64110L1061", "Netflix Inc", 0.03, True),
        ("BTC", "BTC", "US46438F1012", "Bitcoin spot", 0.02, True),
    ]
    return pd.DataFrame(rows, columns=["canonical_id", "ticker", "isins", "name",
                                       "weight", "is_fund"])


def _treatments() -> dict:
    return {"SPY": "equity", "QQQ": "equity", "GLD": "commodity",
            "NFLX": "not_a_fund", "BTC": "crypto"}


def _holdings() -> pd.DataFrame:
    cols = ["fund_id", "constituent_name", "ticker", "isin", "asset_class", "weight_in_fund"]
    rows = [
        ("SPY", "Apple Inc", "AAPL", "US0378331005", "equity", 0.30),
        ("SPY", "Microsoft Corp", "MSFT", "US5949181045", "equity", 0.25),
        ("SPY", "Netflix Inc", "NFLX", "US64110L1061", "equity", 0.10),
        ("SPY", "Alphabet Inc", "GOOG", None, "equity", 0.34),
        ("SPY", "USD Cash", None, None, "cash", 0.01),
        ("QQQ", "Apple Inc", "AAPL", "US0378331005", "equity", 0.40),
        ("QQQ", "Microsoft Corp", "MSFT", "US5949181045", "equity", 0.30),
    ]
    return pd.DataFrame(rows, columns=cols)


def test_resolver_collapses_constituent_onto_direct_holding():
    lookups = lookthrough.build_lookups(_book())
    cid, method = lookthrough.resolve_constituent("Apple Inc", "AAPL", "US0378331005", lookups, {})
    assert cid == "AAPL" and method == "isin"
    cid, method = lookthrough.resolve_constituent("Microsoft", "MSFT", None, lookups, {})
    assert cid == "MSFT" and method == "ticker"
    cid, method = lookthrough.resolve_constituent("Alphabet Inc", "GOOG", None, lookups, {})
    assert cid == "x:GOOG" and method == "synthetic_ticker"


def test_resolver_override_wins():
    lookups = lookthrough.build_lookups(_book())
    cid, method = lookthrough.resolve_constituent("Apple", "AAPL", "US0378331005",
                                                  lookups, {"AAPL": "MSFT"})
    assert cid == "MSFT" and method == "override"


def test_effective_exposure_adds_direct_and_indirect():
    exp, _ = lookthrough.build_effective_exposures(_book(), _holdings(), _treatments())
    e = exp.set_index("canonical_id")
    assert e.loc["AAPL", "true_weight"] == pytest.approx(0.20)
    assert e.loc["AAPL", "indirect_weight"] == pytest.approx(0.10)
    assert e.loc["MSFT", "true_weight"] == pytest.approx(0.13)
    assert e.loc["NFLX", "true_weight"] == pytest.approx(0.05)
    assert e.loc["x:GOOG", "direct_weight"] == pytest.approx(0.0)
    assert e.loc["x:GOOG", "true_weight"] == pytest.approx(0.068)
    assert e.loc["GLD", "true_weight"] == pytest.approx(0.05)
    assert e.loc["BTC", "true_weight"] == pytest.approx(0.02)
    assert "SPY" not in e.index and "QQQ" not in e.index


def test_weight_is_conserved_globally():
    book, holdings = _book(), _holdings()
    exp, _ = lookthrough.build_effective_exposures(book, holdings, _treatments())
    res = lookthrough.reconcile_total(exp, book)
    assert res["reconciled"]
    assert res["total_true_weight"] == pytest.approx(book["weight"].sum())


def test_residual_captures_unattributed_fund_weight():
    exp, recon = lookthrough.build_effective_exposures(_book(), _holdings(), _treatments())
    e = exp.set_index("canonical_id")
    assert e.loc["SPY::cash_other", "true_weight"] == pytest.approx(0.002)
    assert e.loc["QQQ::cash_other", "true_weight"] == pytest.approx(0.03)
    assert bool(e.loc["SPY::cash_other", "is_residual"]) is True


def test_fund_recon_flags_incomplete_download():
    _, recon = lookthrough.build_effective_exposures(_book(), _holdings(), _treatments())
    r = recon.set_index("fund_id")
    assert bool(r.loc["SPY", "reconciled"]) is True
    assert bool(r.loc["QQQ", "reconciled"]) is False


def test_equity_fund_with_no_holdings_stays_as_ballast():
    holdings = _holdings()
    holdings = holdings[holdings["fund_id"] != "SPY"]
    exp, recon = lookthrough.build_effective_exposures(_book(), holdings, _treatments())
    e = exp.set_index("canonical_id")
    assert e.loc["SPY", "true_weight"] == pytest.approx(0.20)
    assert bool(recon.set_index("fund_id").loc["SPY", "looked_through"]) is False
    assert lookthrough.reconcile_total(exp, _book())["reconciled"]


def test_proxy_fund_reuses_source_holdings():
    book = _book()
    book = pd.concat([book, pd.DataFrame([
        ("UBSSPY", "UBSSPY", "IE00B7K93397", "UBS S&P 500", 0.08, True)],
        columns=book.columns)], ignore_index=True)
    treatments = {**_treatments(), "UBSSPY": "equity"}
    holdings = _holdings()
    clone = holdings[holdings["fund_id"] == "SPY"].copy()
    clone["fund_id"] = "UBSSPY"
    holdings = pd.concat([holdings, clone], ignore_index=True)

    exp, recon = lookthrough.build_effective_exposures(
        book, holdings, treatments, proxy_map={"UBSSPY": "SPY"})
    e = exp.set_index("canonical_id")
    assert e.loc["AAPL", "true_weight"] == pytest.approx(0.224)
    assert lookthrough.reconcile_total(exp, book)["reconciled"]
    assert recon.set_index("fund_id").loc["UBSSPY", "proxied_from"] == "SPY"


def test_concentration_report_surfaces_hidden_names():
    exp, _ = lookthrough.build_effective_exposures(_book(), _holdings(), _treatments())
    rep = lookthrough.concentration_report(exp, top_n=10)
    hidden_ids = set(rep["hidden_names"]["canonical_id"])
    assert "x:GOOG" in hidden_ids
    assert not rep["top_true"]["canonical_id"].str.contains("::cash_other").any()
    m = rep["metrics"]
    assert m["n_names_true"] >= m["n_names_declared"]
    assert m["effective_names_true"] is not None


def test_classify_funds_reports_unclassified():
    book = _book()
    treatments, unclassified = lookthrough.classify_funds(
        book, {"SPY": "equity", "GLD": "commodity"})
    assert treatments == {"SPY": "equity", "GLD": "commodity"}
    assert set(unclassified) == {"QQQ", "NFLX", "BTC"}


def test_load_classification_rejects_unknown_treatment(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("equity: [SPY]\nbananas: [QQQ]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown treatment"):
        lookthrough.load_classification(p)


def test_normalize_holdings_scales_percent_and_labels_cash():
    raw = pd.DataFrame({
        "Name": ["Apple Inc", "Microsoft Corp", "USD Cash Collateral"],
        "Ticker": ["AAPL", "MSFT", None],
        "ISIN": ["US0378331005", "US5949181045", None],
        "Asset Class": ["Equity", "Equity", None],
        "Weight (%)": ["30.0", "25.0", "1.0"],
    })
    out = holdings_parse.normalize_holdings(raw, "SPY").set_index("constituent_name")
    assert out.loc["Apple Inc", "weight_in_fund"] == pytest.approx(0.30)
    assert out.loc["Apple Inc", "asset_class"] == "equity"
    assert out.loc["USD Cash Collateral", "asset_class"] == "cash"
    assert list(out.columns) == [c for c in holdings_parse.HOLDING_COLUMNS
                                 if c != "constituent_name"]


def test_normalize_holdings_broad_fund_percent_scale():
    n = 200
    raw = pd.DataFrame({
        "Name": [f"Stock {i}" for i in range(n)],
        "Ticker": [f"T{i}" for i in range(n)],
        "Weight (%)": [0.5] * n,
    })
    out = holdings_parse.normalize_holdings(raw, "IWM")
    assert out["weight_in_fund"].sum() == pytest.approx(1.0)
    assert out["weight_in_fund"].iloc[0] == pytest.approx(0.005)


def test_normalize_holdings_requires_weight_column():
    raw = pd.DataFrame({"Name": ["Apple"], "Ticker": ["AAPL"]})
    with pytest.raises(ValueError, match="no weight column"):
        holdings_parse.normalize_holdings(raw, "SPY")
