"""Phase 7 — risk attribution & stress.

Turns the covariance map into a risk budget and stresses it:

  risk budget   — portfolio vol split into each holding's marginal (MCTR) and component
                  (CCTR) contribution, aggregated to Phase-5 clusters; effective # of
                  risk bets = 1 / Σ(pct_contribution²).
  stress        — recompute portfolio vol after swapping in a stressed CORRELATION regime
                  (keeping current vols/weights, isolating the diversification breakdown):
                  the 2020 / 2022 episode correlations, and an all-correlations->0.9 tail.

Covariance Σ = D·C·D (D = diag of daily vols, C = the cleaned correlation). Risk is the
one place we DO use the market-inclusive (cleaned, not detoned) matrix — the market is a
real risk source. The matrix is never inverted; only the quadratic form wᵀΣw is used.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("cov.analytics.stress")

ANNUALIZE = np.sqrt(252.0)


# --- covariance & portfolio vol ----------------------------------------------
def cov_from_corr(corr: pd.DataFrame, vols: pd.Series) -> pd.DataFrame:
    """Σ = D·C·D from a correlation matrix and a vector of (daily) vols."""
    d = vols.reindex(corr.index).to_numpy()
    return pd.DataFrame(np.outer(d, d) * corr.to_numpy(), index=corr.index, columns=corr.columns)


def portfolio_vol(cov: pd.DataFrame, weights: pd.Series) -> float:
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    return float(np.sqrt(max(w @ cov.to_numpy() @ w, 0.0)))


# --- risk decomposition (Euler / MCTR-CCTR) ----------------------------------
def risk_contributions(cov: pd.DataFrame, weights: pd.Series) -> tuple[pd.DataFrame, float]:
    """Per-holding marginal (MCTR) and component (CCTR) risk. ΣCCTR = portfolio vol.

    MCTR_i = (Σw)_i / σ_p ; CCTR_i = w_i·MCTR_i ; pct_risk_i = CCTR_i / σ_p (sums to 1).
    """
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    sig = float(np.sqrt(max(w @ cov.to_numpy() @ w, 0.0)))
    if sig <= 0:
        raise ValueError("degenerate portfolio (zero variance)")
    mctr = cov.to_numpy() @ w / sig
    cctr = w * mctr
    out = pd.DataFrame({"weight": w, "mctr": mctr, "cctr": cctr, "pct_risk": cctr / sig},
                       index=cov.index).sort_values("pct_risk", ascending=False)
    return out, sig


def effective_risk_bets(pct_risk: pd.Series) -> float:
    """1 / Σ(pct²): the effective number of equal-risk bets (N if perfectly spread)."""
    return float(1.0 / np.square(pct_risk.to_numpy()).sum())


def cluster_risk(contrib: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """Aggregate pct-risk and weight to Phase-5 clusters."""
    df = contrib.join(clusters.rename("cluster"))
    g = df.groupby("cluster").agg(weight=("weight", "sum"), pct_risk=("pct_risk", "sum"),
                                  n=("weight", "size"))
    g["risk_vs_weight"] = (g["pct_risk"] / g["weight"]).round(3)   # >1 = punches above its weight
    return g.sort_values("pct_risk", ascending=False)


# --- stressed correlation regimes --------------------------------------------
def scenario_all_to(corr: pd.DataFrame, rho: float) -> pd.DataFrame:
    """Force every off-diagonal correlation to ``rho`` (the crisis-convergence tail)."""
    c = np.full(corr.shape, float(rho))
    np.fill_diagonal(c, 1.0)
    return pd.DataFrame(c, index=corr.index, columns=corr.columns)


def episode_correlation(returns: pd.DataFrame, start: str, end: str,
                        baseline: pd.DataFrame, min_obs: int = 8) -> pd.DataFrame:
    """Sample correlation during an episode window, aligned to ``baseline``; pairs with
    too little overlap fall back to the baseline correlation (never NaN)."""
    idx = pd.to_datetime(returns.index)
    win = returns.loc[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    cols = [c for c in baseline.columns if c in win.columns]
    ec = win[cols].corr(min_periods=min_obs)
    ec = ec.reindex(index=baseline.index, columns=baseline.columns)
    ec = ec.fillna(baseline)
    vals = ec.to_numpy().copy(); np.fill_diagonal(vals, 1.0)
    return pd.DataFrame(vals, index=baseline.index, columns=baseline.columns)


def avg_offdiag(corr: pd.DataFrame) -> float:
    n = corr.shape[0]
    return float((corr.to_numpy().sum() - n) / (n * (n - 1))) if n > 1 else float("nan")


def stress_table(baseline_corr: pd.DataFrame, vols: pd.Series, weights: pd.Series,
                 returns: pd.DataFrame, episodes: dict, all_to: float) -> pd.DataFrame:
    """Portfolio vol (annualized) under baseline + each episode regime + the all->rho tail.

    Correlation is stressed; vols and weights are held current, so the delta is the pure
    diversification-breakdown effect.
    """
    rows = []

    def _vol(corr, label):
        v = portfolio_vol(cov_from_corr(corr, vols), weights) * ANNUALIZE
        rows.append({"scenario": label, "avg_corr": round(avg_offdiag(corr), 4),
                     "annual_vol": round(v, 4)})
        return v

    base = _vol(baseline_corr, "baseline")
    for name, ep in episodes.items():
        ec = episode_correlation(returns, str(ep["start"]), str(ep["end"]), baseline_corr)
        _vol(ec, name)
    _vol(scenario_all_to(baseline_corr, all_to), f"all_corr_to_{all_to}")

    tab = pd.DataFrame(rows)
    tab["vs_baseline"] = (tab["annual_vol"] / base).round(3)   # x-fold vs baseline
    return tab
