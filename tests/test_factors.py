"""Phase 6 tests — factor panel, holding regressions, portfolio exposure, factor naming.

Offline: a FakeFactorFetcher supplies synthetic macro series, and holdings are built as
known linear combinations of factors so the regression must recover the planted betas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics import factors as fx


def _internal_returns(t=500, seed=0):
    """A returns frame carrying the internal factor proxies + a couple of holdings."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=t)
    acwi = rng.standard_normal(t) * 0.01
    vong = rng.standard_normal(t) * 0.01
    qdiv = rng.standard_normal(t) * 0.01
    btc = rng.standard_normal(t) * 0.03
    return pd.DataFrame({"ACWI": acwi, "VONG": vong, "QDIV": qdiv, "BTC": btc}, index=idx)


class FakeFactorFetcher(fx.FactorFetcher):
    """Deterministic macro levels, so diff/logret transforms produce finite series."""
    def fetch(self, codes, start, end):
        idx = pd.bdate_range("2021-01-01", periods=500)
        rng = np.random.default_rng(1)
        return pd.DataFrame({name: 100 + np.cumsum(rng.standard_normal(500) * 0.1)
                             for name in codes}, index=idx)


# --- factor panel ------------------------------------------------------------
def test_build_panel_internal_and_macro():
    r = _internal_returns()
    panel, sources = fx.build_factor_panel(r, FakeFactorFetcher(), "2021-01-01", "2022-12-31")
    assert {"MKT", "GROWTH_VALUE", "CRYPTO", "RATES", "CREDIT", "USD", "OIL"}.issubset(panel.columns)
    assert sources["MKT"] == "ACWI" and sources["GROWTH_VALUE"] == "VONG-QDIV"
    # GROWTH_VALUE is exactly VONG - QDIV
    assert np.allclose(panel["GROWTH_VALUE"].dropna(), (r["VONG"] - r["QDIV"]).dropna())


def test_panel_skips_missing_proxy():
    r = _internal_returns().drop(columns=["BTC"])
    panel, sources = fx.build_factor_panel(r, None, "2021-01-01", "2022-12-31")
    assert "CRYPTO" not in panel.columns          # proxy absent -> factor skipped, not faked
    assert "MKT" in panel.columns


# --- regression recovers planted betas ---------------------------------------
def test_regress_recovers_betas():
    r = _internal_returns()
    panel, _ = fx.build_factor_panel(r, None, "2021-01-01", "2022-12-31")
    rng = np.random.default_rng(2)
    # a holding that is 2*MKT - 1*CRYPTO + small noise
    y = 2.0 * panel["MKT"] - 1.0 * panel["CRYPTO"] + rng.standard_normal(len(panel)) * 1e-4
    betas, r2 = fx.regress_holdings(pd.DataFrame({"H": y}), panel, min_obs=100)
    assert betas.loc["H", "MKT"] == pytest.approx(2.0, abs=0.05)
    assert betas.loc["H", "CRYPTO"] == pytest.approx(-1.0, abs=0.05)
    assert r2["H"] > 0.99


# --- portfolio exposure = weighted sum of betas ------------------------------
def test_portfolio_exposure_weighted_sum():
    betas = pd.DataFrame({"MKT": [1.0, 0.0], "USD": [0.0, -2.0]}, index=["A", "B"])
    weights = pd.Series({"A": 0.75, "B": 0.25})
    exp = fx.portfolio_exposure(betas, weights)
    assert exp.loc["MKT", "net_beta"] == pytest.approx(0.75)     # 0.75*1 + 0.25*0
    assert exp.loc["USD", "net_beta"] == pytest.approx(-0.5)     # 0.25*-2
    # sorted by absolute magnitude -> USD (0.5) before MKT (0.75)? no: 0.75 > 0.5
    assert list(exp.index) == ["MKT", "USD"]


# --- naming a PCA factor -----------------------------------------------------
def test_name_factor_identifies_dominant():
    r = _internal_returns()
    panel, _ = fx.build_factor_panel(r, None, "2021-01-01", "2022-12-31")
    pca = pd.DataFrame({"F1": panel["MKT"]}, index=panel.index)   # F1 IS the market
    named = fx.name_factors(pca, panel, top=3)
    row = named[named["pca_factor"] == "F1"].iloc[0]
    assert "MKT" in row["economic_name"]
    assert row["r2"] > 0.99
