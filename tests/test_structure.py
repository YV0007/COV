"""Phase 5 tests — correlation distance, clustering, sector confusion, absorption ratio.

Offline and deterministic: synthetic block-factor returns (K clean blocks) so we can
assert clustering recovers the planted groups and the sector-agreement metrics behave.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics import structure as sx


def _block_returns(seed=0, t=800, blocks=3, per=8):
    """t days; `blocks` groups of `per` assets, each group driven by a shared factor."""
    rng = np.random.default_rng(seed)
    fac = rng.standard_normal((t, blocks))
    cols, arrs, truth = [], [], []
    for bi in range(blocks):
        for j in range(per):
            arrs.append(fac[:, bi] + 0.4 * rng.standard_normal(t))
            cols.append(f"B{bi}_{j}")
            truth.append(bi)
    df = pd.DataFrame(np.array(arrs).T, columns=cols,
                      index=pd.bdate_range("2020-01-01", periods=t))
    return df, dict(zip(cols, truth))


# --- correlation distance ----------------------------------------------------
def test_correlation_distance_metric():
    corr = pd.DataFrame([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        index=list("abc"), columns=list("abc"))
    d = sx.correlation_distance(corr)
    assert d.loc["a", "b"] == pytest.approx(0.0)          # rho=1 -> 0
    assert d.loc["a", "c"] == pytest.approx(np.sqrt(2))   # rho=0 -> sqrt2
    assert np.allclose(d.values, d.values.T)              # symmetric
    assert np.allclose(np.diag(d.values), 0.0)


# --- clustering recovers planted blocks --------------------------------------
def test_clustering_recovers_blocks():
    df, truth = _block_returns(blocks=3, per=8)
    corr = df.corr()
    z, clusters = sx.hierarchical_clusters(corr, method="ward", k=3)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score([truth[c] for c in clusters.index], clusters.values)
    assert ari == pytest.approx(1.0)                      # clean blocks -> perfect recovery
    assert clusters.nunique() == 3


# --- PCA factor table --------------------------------------------------------
def test_pca_factors_lists_drivers():
    df, truth = _block_returns(blocks=3, per=8)
    corr = df.corr()
    sectors = {c: f"S{truth[c]}" for c in corr.index}
    ft = sx.pca_factors(corr, sectors, n=3, top_k=5)
    assert len(ft) == 3
    assert {"factor", "variance_share", "dominant_sector", "drivers"}.issubset(ft.columns)
    assert ft["variance_share"].iloc[0] >= ft["variance_share"].iloc[1]   # sorted desc


# --- sector confusion --------------------------------------------------------
def test_sector_confusion_perfect_alignment():
    df, truth = _block_returns(blocks=3, per=8)
    _, clusters = sx.hierarchical_clusters(df.corr(), method="ward", k=3)
    sectors = {c: f"S{truth[c]}" for c in clusters.index}     # sector == true block
    conf, stats, per = sx.sector_confusion(clusters, sectors)
    assert stats["adjusted_rand"] == pytest.approx(1.0)
    assert stats["n_divergent"] == 0                          # nothing mislabeled
    assert conf.shape == (3, 3)


def test_sector_confusion_flags_a_mislabel():
    df, truth = _block_returns(blocks=3, per=8)
    _, clusters = sx.hierarchical_clusters(df.corr(), method="ward", k=3)
    sectors = {c: f"S{truth[c]}" for c in clusters.index}
    mis = clusters.index[0]                                   # a block-0 holding
    other = (truth[mis] + 1) % 3
    sectors[mis] = f"S{other}"     # label it as ANOTHER block's sector (whose majority
    #                                cluster differs) -> it becomes behaviourally divergent
    _, stats, per = sx.sector_confusion(clusters, sectors)
    assert stats["n_divergent"] >= 1


# --- absorption ratio --------------------------------------------------------
def test_absorption_ratio_bounds():
    assert sx.absorption_ratio(np.ones(20), top_n=5) == pytest.approx(0.25)   # identity: 5/20
    dominant = np.array([100.0] + [0.1] * 19)
    assert sx.absorption_ratio(dominant, top_n=5) > 0.9                       # one big factor


def test_rolling_absorption_series():
    df, _ = _block_returns(t=600)
    s = sx.rolling_absorption(df, top_n=5, window=120, step=40)
    assert {"date", "absorption_ratio"}.issubset(s.columns)
    assert (s["absorption_ratio"].between(0, 1)).all()
    assert len(s) >= 5
