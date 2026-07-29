"""Phase 4 — the covariance / correlation engine (core estimators & cleaning).

Turns the Phase 2 returns matrix into a *trustworthy* correlation matrix by peeling
noise out of it, in the order the intuition describes:

  1. sample correlation           — the blurry photo
  2. Marchenko-Pastur band        — where the noise floor is (q = N/T)
  3. Ledoit-Wolf shrinkage        — one global linear pull toward a stable prior
  4. eigenvalue cleaning (RIE/clip)— per-eigenvalue surgery: keep directions, fix sizes
  5. detoning                     — subtract the market mode so structure is visible
  8. metadata                     — condition number, effective rank, factor count

Everything operates on eigenvalues/eigenvectors of a symmetric matrix, so the maths is
just ``np.linalg.eigh`` plus Random-Matrix-Theory formulas — no matrix is ever inverted
for construction (this is a measurement instrument, per the project's remit).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("cov.engine.covariance")


# --- data prep ----------------------------------------------------------------
def prepare_block(returns: pd.DataFrame, min_coverage: float = 0.9,
                  min_row_coverage: float = 0.5) -> pd.DataFrame:
    """Reduce the (dates × instruments) returns to a dense, no-NaN estimation block.

    Estimators (Ledoit-Wolf, eigh) need a gap-free matrix, but a naive ``dropna`` empties
    the panel: crypto trades weekends and exchanges keep different holidays, so almost no
    single day has *every* instrument present. Instead:

      1. keep instruments present on >= ``min_coverage`` of days (drops short-history names);
      2. drop sparse rows — weekends where only crypto trades (< ``min_row_coverage`` present);
      3. trim the leading period until >=90% of kept names are live;
      4. fill the residual scattered holiday gaps with 0 (a non-trading day = no return).

    The excluded short-history names are reported as reduced coverage, never silently kept.
    """
    if returns is None or returns.empty:
        return pd.DataFrame()
    # 1. drop sparse rows FIRST — weekend rows exist because crypto trades 7 days/week;
    #    on them almost every equity is NaN. Computing coverage before this would penalise
    #    every equity to ~5/7 and wrongly drop them all.
    sub = returns[returns.notna().mean(axis=1) >= min_row_coverage]
    # 2. now keep instruments present on >= min_coverage of the remaining business days
    coverage = sub.notna().mean(axis=0)
    sub = sub[coverage[coverage >= min_coverage].index]
    # 3. trim the leading period until >=90% of kept names are live, then fill holiday gaps
    dense = sub.notna().mean(axis=1) >= 0.9
    if dense.any():
        sub = sub.loc[dense.idxmax():]
    return sub.fillna(0.0)                                      # holiday gap = 0 return


# --- eigen utilities ----------------------------------------------------------
def eigidecompose(corr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues desc, eigenvectors) of a symmetric correlation matrix."""
    vals, vecs = np.linalg.eigh(np.asarray(corr, dtype=float))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def _reassemble(vals: np.ndarray, vecs: np.ndarray, labels) -> pd.DataFrame:
    """Rebuild a matrix from (possibly modified) eigenvalues and renormalize to unit
    diagonal — so the result is a proper correlation matrix again."""
    mat = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(mat), 1e-12, None))
    out = mat / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return pd.DataFrame(out, index=labels, columns=labels)


def mp_edges(n: int, t: int) -> tuple[float, float, float]:
    """Marchenko-Pastur band edges (λ-, λ+) and the aspect ratio q = N/T."""
    q = n / t
    sq = np.sqrt(q)
    return (1 - sq) ** 2, (1 + sq) ** 2, q


def count_factors(vals: np.ndarray, lam_plus: float) -> int:
    """Number of eigenvalues above the MP noise ceiling = the significant factors."""
    return int((vals > lam_plus).sum())


# --- Ledoit-Wolf shrinkage ----------------------------------------------------
def ledoit_wolf_corr(block: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrunk correlation + the shrinkage intensity δ (0..1).

    δ is the estimator's own data-quality signal: high δ = "the sample was noisy, I
    leaned on the prior". Uses scikit-learn's analytic optimal δ (no tuning).
    """
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(block.values)
    cov = lw.covariance_
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=block.columns, columns=block.columns), float(lw.shrinkage_)


# --- eigenvalue cleaning ------------------------------------------------------
def clip_clean(corr: pd.DataFrame, lam_plus: float) -> pd.DataFrame:
    """Laloux clipping: flatten every eigenvalue below the MP ceiling to their average
    (trace preserved); keep the ones above. The crude-but-robust cleaner."""
    vals, vecs = eigidecompose(corr)
    noise = vals < lam_plus
    cleaned = vals.astype(float).copy()
    if noise.any():
        cleaned[noise] = vals[noise].mean()
    cleaned *= vals.sum() / cleaned.sum()          # preserve total variance
    return _reassemble(cleaned, vecs, corr.index)


def rie_clean(corr: pd.DataFrame, q: float) -> pd.DataFrame:
    """Rotationally-invariant estimator (Bun-Bouchaud-Potters): keep eigenvectors, and
    replace each eigenvalue by its RMT-optimal value via the sample Stieltjes transform.

        ξ_k = λ_k / |1 − q + q·λ_k·s(λ_k − iη)|²,   s(z) = (1/N) Σ_j 1/(z − λ_j)

    Big (real-factor) eigenvalues are barely touched (s(λ)≈1/λ ⇒ denominator≈1); the
    noise bulk is compressed toward its centre. Trace is rescaled back to preserve
    total variance. η ~ N^{-1/2} is the standard regularization.
    """
    vals, vecs = eigidecompose(corr)
    n = len(vals)
    eta = n ** -0.5
    z = vals - 1j * eta
    s = (1.0 / n) * (1.0 / (z[:, None] - vals[None, :])).sum(axis=1)
    xi = np.real(vals / np.abs(1.0 - q + q * vals * s) ** 2)
    xi = np.clip(xi, 1e-8, None)
    xi *= vals.sum() / xi.sum()                    # preserve total variance
    return _reassemble(xi, vecs, corr.index)


def clean(corr: pd.DataFrame, n: int, t: int, method: str = "rie") -> pd.DataFrame:
    """Dispatch to the configured cleaner (``rie`` | ``clip`` | ``none``)."""
    lam_minus, lam_plus, q = mp_edges(n, t)
    if method == "none":
        return corr
    if method == "clip":
        return clip_clean(corr, lam_plus)
    return rie_clean(corr, q)


# --- detoning -----------------------------------------------------------------
def detone(corr: pd.DataFrame, n_market: int = 1) -> tuple[pd.DataFrame, float]:
    """Remove the top ``n_market`` eigenmode(s) — the market tide — and renormalize.

    Returns (detoned correlation, market variance share). The removed mode is the single
    biggest 'bet'; stripping it lets sectors/styles surface for Phase 5 clustering.
    """
    vals, vecs = eigidecompose(corr)
    market_share = float(vals[:n_market].sum() / vals.sum())
    kept = vals.astype(float).copy()
    kept[:n_market] = 0.0
    return _reassemble(kept, vecs, corr.index), market_share


# --- metadata (honesty labels) ------------------------------------------------
def matrix_metadata(corr: pd.DataFrame) -> dict:
    """Condition number and effective rank (≈ number of independent bets).

    effective_rank = (Σλ)² / Σλ²  — the participation ratio: N for a perfectly diverse
    book, small when one factor dominates. condition_number = λ_max / λ_min (positive
    eigenvalues) — numerical stability of the estimate.
    """
    vals, _ = eigidecompose(corr)
    pos = vals[vals > 1e-10]
    cond = float(pos.max() / pos.min()) if len(pos) else float("nan")
    eff_rank = float(vals.sum() ** 2 / np.square(vals).sum())
    return {"condition_number": round(cond, 2), "effective_rank": round(eff_rank, 2)}
