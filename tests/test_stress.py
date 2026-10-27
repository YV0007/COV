# Phase 7 tests — covariance, MCTR/CCTR risk decomposition, effective bets, stress

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics import stress as ss


def _corr():
    return pd.DataFrame([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        index=list("abc"), columns=list("abc"))


def test_cov_from_corr_and_vol():
    vols = pd.Series({"a": 0.1, "b": 0.2, "c": 0.3})
    cov = ss.cov_from_corr(_corr(), vols)
    assert cov.loc["a", "a"] == pytest.approx(0.01)
    assert cov.loc["a", "b"] == pytest.approx(0.5 * 0.1 * 0.2)
    w = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    expected = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()))
    assert ss.portfolio_vol(cov, w) == pytest.approx(expected)


def test_risk_contributions_euler_identity():
    vols = pd.Series({"a": 0.1, "b": 0.2, "c": 0.3})
    cov = ss.cov_from_corr(_corr(), vols)
    w = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    contrib, sig = ss.risk_contributions(cov, w)
    assert contrib["cctr"].sum() == pytest.approx(sig)
    assert contrib["pct_risk"].sum() == pytest.approx(1.0)


def test_effective_risk_bets():
    assert ss.effective_risk_bets(pd.Series([0.25] * 4)) == pytest.approx(4.0)
    concentrated = pd.Series([0.9, 0.05, 0.03, 0.02])
    assert ss.effective_risk_bets(concentrated) < 1.5


def test_scenario_all_to():
    c = ss.scenario_all_to(_corr(), 0.9)
    assert np.allclose(np.diag(c.values), 1.0)
    off = c.values[~np.eye(3, dtype=bool)]
    assert np.allclose(off, 0.9)


def _returns(t=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=t)
    return pd.DataFrame(rng.standard_normal((t, 3)) * 0.01, index=idx, columns=list("abc"))


def test_episode_correlation_no_nan_and_aligned():
    r = _returns()
    base = _corr()
    ec = ss.episode_correlation(r, "2020-02-01", "2020-03-15", base)
    assert list(ec.index) == list(base.index)
    assert not ec.isna().any().any()
    assert np.allclose(np.diag(ec.values), 1.0)


def test_stress_table_tail_raises_vol():
    r = _returns()
    base = _corr()
    vols = pd.Series({"a": 0.1, "b": 0.2, "c": 0.3})
    w = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    episodes = {"ep": {"start": "2020-02-01", "end": "2020-03-15"}}
    tab = ss.stress_table(base, vols, w, r, episodes, all_to=0.9)
    assert set(tab["scenario"]) == {"baseline", "ep", "all_corr_to_0.9"}
    base_v = tab.loc[tab["scenario"] == "baseline", "annual_vol"].iloc[0]
    tail_v = tab.loc[tab["scenario"] == "all_corr_to_0.9", "annual_vol"].iloc[0]
    assert tail_v > base_v
    assert tab.loc[tab["scenario"] == "baseline", "vs_baseline"].iloc[0] == pytest.approx(1.0)
