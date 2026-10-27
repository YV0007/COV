# Phase 4 — the covariance / correlation engine (core estimators & cleaning)

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("cov.engine.covariance")


def prepare_block(returns: pd.DataFrame, min_coverage: float = 0.9,
                  min_row_coverage: float = 0.5) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    sub = returns[returns.notna().mean(axis=1) >= min_row_coverage]
    coverage = sub.notna().mean(axis=0)
    sub = sub[coverage[coverage >= min_coverage].index]
    dense = sub.notna().mean(axis=1) >= 0.9
    if dense.any():
        sub = sub.loc[dense.idxmax():]
    return sub.fillna(0.0)


def eigidecompose(corr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(np.asarray(corr, dtype=float))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def _reassemble(vals: np.ndarray, vecs: np.ndarray, labels) -> pd.DataFrame:
    mat = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(mat), 1e-12, None))
    out = mat / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return pd.DataFrame(out, index=labels, columns=labels)


def mp_edges(n: int, t: int) -> tuple[float, float, float]:
    q = n / t
    sq = np.sqrt(q)
    return (1 - sq) ** 2, (1 + sq) ** 2, q


def count_factors(vals: np.ndarray, lam_plus: float) -> int:
    return int((vals > lam_plus).sum())


def ledoit_wolf_corr(block: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(block.values)
    cov = lw.covariance_
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=block.columns, columns=block.columns), float(lw.shrinkage_)


def clip_clean(corr: pd.DataFrame, lam_plus: float) -> pd.DataFrame:
    vals, vecs = eigidecompose(corr)
    noise = vals < lam_plus
    cleaned = vals.astype(float).copy()
    if noise.any():
        cleaned[noise] = vals[noise].mean()
    cleaned *= vals.sum() / cleaned.sum()
    return _reassemble(cleaned, vecs, corr.index)


def rie_clean(corr: pd.DataFrame, q: float) -> pd.DataFrame:
    vals, vecs = eigidecompose(corr)
    n = len(vals)
    eta = n ** -0.5
    z = vals - 1j * eta
    s = (1.0 / n) * (1.0 / (z[:, None] - vals[None, :])).sum(axis=1)
    xi = np.real(vals / np.abs(1.0 - q + q * vals * s) ** 2)
    xi = np.clip(xi, 1e-8, None)
    xi *= vals.sum() / xi.sum()
    return _reassemble(xi, vecs, corr.index)


def clean(corr: pd.DataFrame, n: int, t: int, method: str = "rie") -> pd.DataFrame:
    lam_minus, lam_plus, q = mp_edges(n, t)
    if method == "none":
        return corr
    if method == "clip":
        return clip_clean(corr, lam_plus)
    return rie_clean(corr, q)


def detone(corr: pd.DataFrame, n_market: int = 1) -> tuple[pd.DataFrame, float]:
    vals, vecs = eigidecompose(corr)
    market_share = float(vals[:n_market].sum() / vals.sum())
    kept = vals.astype(float).copy()
    kept[:n_market] = 0.0
    return _reassemble(kept, vecs, corr.index), market_share


def matrix_metadata(corr: pd.DataFrame) -> dict:
    vals, _ = eigidecompose(corr)
    pos = vals[vals > 1e-10]
    cond = float(pos.max() / pos.min()) if len(pos) else float("nan")
    eff_rank = float(vals.sum() ** 2 / np.square(vals).sum())
    return {"condition_number": round(cond, 2), "effective_rank": round(eff_rank, 2)}
