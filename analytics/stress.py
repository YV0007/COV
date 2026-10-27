# Phase 7 — risk attribution & stress

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("cov.analytics.stress")

ANNUALIZE = np.sqrt(252.0)


def cov_from_corr(corr: pd.DataFrame, vols: pd.Series) -> pd.DataFrame:
    d = vols.reindex(corr.index).to_numpy()
    return pd.DataFrame(np.outer(d, d) * corr.to_numpy(), index=corr.index, columns=corr.columns)


def portfolio_vol(cov: pd.DataFrame, weights: pd.Series) -> float:
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    return float(np.sqrt(max(w @ cov.to_numpy() @ w, 0.0)))


def risk_contributions(cov: pd.DataFrame, weights: pd.Series) -> tuple[pd.DataFrame, float]:
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
    return float(1.0 / np.square(pct_risk.to_numpy()).sum())


def cluster_risk(contrib: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    df = contrib.join(clusters.rename("cluster"))
    g = df.groupby("cluster").agg(weight=("weight", "sum"), pct_risk=("pct_risk", "sum"),
                                  n=("weight", "size"))
    g["risk_vs_weight"] = (g["pct_risk"] / g["weight"]).round(3)
    return g.sort_values("pct_risk", ascending=False)


def scenario_all_to(corr: pd.DataFrame, rho: float) -> pd.DataFrame:
    c = np.full(corr.shape, float(rho))
    np.fill_diagonal(c, 1.0)
    return pd.DataFrame(c, index=corr.index, columns=corr.columns)


def episode_correlation(returns: pd.DataFrame, start: str, end: str,
                        baseline: pd.DataFrame, min_obs: int = 8) -> pd.DataFrame:
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
    tab["vs_baseline"] = (tab["annual_vol"] / base).round(3)
    return tab
