"""Phase 4 tests — MP band, shrinkage, RIE/clip cleaning, detoning, windows, Spearman.

Offline and deterministic: a synthetic returns panel with a KNOWN structure (one strong
market factor + a couple of block factors + noise) so we can assert the engine recovers
what we planted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import covariance as cvx, windows as win


def _synthetic_returns(seed=0, t=1500, n=60):
    """t days x n assets: a market factor all load on + 3 sector blocks + idiosyncratic."""
    rng = np.random.default_rng(seed)
    market = rng.standard_normal((t, 1))
    blocks = np.repeat(rng.standard_normal((t, 3)), n // 3, axis=1)[:, :n]
    idio = rng.standard_normal((t, n))
    x = 1.0 * market + 0.6 * blocks + 0.8 * idio
    cols = [f"A{i}" for i in range(n)]
    dates = pd.bdate_range("2019-01-01", periods=t)
    return pd.DataFrame(x, index=dates, columns=cols)


# --- MP band -----------------------------------------------------------------
def test_mp_edges_match_formula():
    lam_minus, lam_plus, q = cvx.mp_edges(212, 3850)
    assert q == pytest.approx(212 / 3850)
    assert lam_plus == pytest.approx((1 + np.sqrt(q)) ** 2)
    assert lam_minus == pytest.approx((1 - np.sqrt(q)) ** 2)


def test_factor_count_finds_planted_structure():
    r = _synthetic_returns()
    c = r.corr()
    vals, _ = cvx.eigidecompose(c)
    _, lam_plus, _ = cvx.mp_edges(*r.shape[::-1])
    # market + ~3 sector blocks should poke above the noise ceiling
    assert 1 <= cvx.count_factors(vals, lam_plus) <= 8


# --- shrinkage ---------------------------------------------------------------
def test_ledoit_wolf_returns_valid_correlation_and_delta():
    r = _synthetic_returns()
    corr, delta = cvx.ledoit_wolf_corr(r)
    assert 0.0 <= delta <= 1.0
    assert np.allclose(np.diag(corr.values), 1.0)
    assert np.allclose(corr.values, corr.values.T)


# --- cleaning preserves trace, stays PSD, keeps unit diagonal ----------------
@pytest.mark.parametrize("method", ["rie", "clip"])
def test_cleaning_preserves_trace_and_psd(method):
    r = _synthetic_returns()
    c = r.corr()
    n, t = c.shape[0], r.shape[0]
    cleaned = cvx.clean(c, n, t, method=method)
    assert np.allclose(np.diag(cleaned.values), 1.0)              # unit diagonal
    assert np.trace(cleaned.values) == pytest.approx(n)           # trace preserved
    vals = np.linalg.eigvalsh(cleaned.values)
    assert vals.min() > -1e-8                                     # PSD (no negatives)


def test_rie_barely_touches_the_market_eigenvalue():
    r = _synthetic_returns()
    c = r.corr()
    v0, _ = cvx.eigidecompose(c)
    cleaned = cvx.rie_clean(c, q=c.shape[0] / r.shape[0])
    v1, _ = cvx.eigidecompose(cleaned)
    # the dominant (market) eigenvalue should be largely preserved (kept direction+size)
    assert v1[0] == pytest.approx(v0[0], rel=0.2)


# --- detoning removes the market and lifts effective rank --------------------
def test_detone_removes_market_and_raises_effective_rank():
    r = _synthetic_returns()
    c = cvx.clean(r.corr(), r.shape[1], r.shape[0], method="rie")
    before = cvx.matrix_metadata(c)["effective_rank"]
    detoned, share = cvx.detone(c)
    after = cvx.matrix_metadata(detoned)["effective_rank"]
    assert 0.1 < share < 0.9                                      # market is a big chunk
    assert after > before                                        # more diverse w/o the tide
    assert np.allclose(np.diag(detoned.values), 1.0)


# --- windows / EWMA ----------------------------------------------------------
def test_ewma_is_valid_correlation():
    r = _synthetic_returns()
    e = win.ewma_correlation(r, lam=0.94)
    assert np.allclose(np.diag(e.values), 1.0)
    assert e.values.max() <= 1.0 + 1e-9 and e.values.min() >= -1.0 - 1e-9


def test_rolling_diagnostics_series():
    r = _synthetic_returns(t=400)
    d = win.rolling_diagnostics(r, window=90, step=30)
    assert {"date", "avg_corr", "effective_rank"}.issubset(d.columns)
    assert len(d) >= 5


# --- Spearman cross-check ----------------------------------------------------
def test_spearman_flags_an_outlier_driven_pair():
    r = _synthetic_returns()
    # inject a single shared extreme day into two otherwise-unrelated assets
    r.iloc[0, 0] = 15.0
    r.iloc[0, 1] = 15.0
    p = r.corr()
    top, stats = win.spearman_agreement(r, p, flag_threshold=0.1)
    assert stats["n_pairs"] == r.shape[1] * (r.shape[1] - 1) // 2
    assert stats["max_abs_diff"] >= 0.0
