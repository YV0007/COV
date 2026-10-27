# Phase 4 — measuring at several speeds, the robustness ruler, and the MP plot

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("cov.engine.windows")


def ewma_correlation(block: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    x = block.values
    t = len(x)
    ages = np.arange(t)[::-1]
    w = (1 - lam) * lam ** ages
    w /= w.sum()
    mu = (w[:, None] * x).sum(axis=0)
    xc = x - mu
    cov = (xc * w[:, None]).T @ xc
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=block.columns, columns=block.columns)


def rolling_diagnostics(block: pd.DataFrame, window: int = 90, step: int = 21) -> pd.DataFrame:
    idx = block.index
    rows = []
    for end in range(window, len(block) + 1, step):
        sub = block.iloc[end - window:end]
        sub = sub.loc[:, sub.std(axis=0) > 0]
        n = sub.shape[1]
        if n < 2:
            continue
        c = np.nan_to_num(sub.corr().values, nan=0.0)
        np.fill_diagonal(c, 1.0)
        offdiag = (c.sum() - n) / (n * (n - 1))
        vals = np.linalg.eigvalsh(c)
        eff = float(vals.sum() ** 2 / np.square(vals).sum())
        rows.append({"date": idx[end - 1], "avg_corr": round(offdiag, 6),
                     "effective_rank": round(eff, 3), "n_instruments": n})
    return pd.DataFrame(rows)


def spearman_agreement(block: pd.DataFrame, pearson: pd.DataFrame,
                       flag_threshold: float = 0.25) -> tuple[pd.DataFrame, dict]:
    spear = block.corr(method="spearman")
    diff = (pearson - spear).abs()
    iu = np.triu_indices(len(diff), k=1)
    pairs = pd.DataFrame({
        "i": diff.index.values[iu[0]],
        "j": diff.columns.values[iu[1]],
        "pearson": pearson.values[iu],
        "spearman": spear.values[iu],
        "abs_diff": diff.values[iu],
    })
    n_flag = int((pairs["abs_diff"] > flag_threshold).sum())
    stats = {"mean_abs_diff": round(float(pairs["abs_diff"].mean()), 4),
             "max_abs_diff": round(float(pairs["abs_diff"].max()), 4),
             "n_flagged_pairs": n_flag, "n_pairs": len(pairs)}
    top = pairs.nlargest(20, "abs_diff").reset_index(drop=True)
    return top, stats


def mp_density(lam: np.ndarray, q: float) -> np.ndarray:
    lam_minus, lam_plus = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    inside = (lam >= lam_minus) & (lam <= lam_plus)
    dens = np.zeros_like(lam)
    dens[inside] = np.sqrt((lam_plus - lam[inside]) * (lam[inside] - lam_minus)) / (
        2 * np.pi * q * lam[inside])
    return dens


def plot_mp_spectrum(vals: np.ndarray, q: float, out_path: Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam_minus, lam_plus = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    bulk = vals[vals <= lam_plus]
    outliers = vals[vals > lam_plus]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(bulk, bins=60, density=True, alpha=0.55, color="#4C72B0", label="empirical (bulk)")
    grid = np.linspace(max(lam_minus, 1e-6), lam_plus, 400)
    ax.plot(grid, mp_density(grid, q), color="#C44E52", lw=2, label="MP pure-noise density")
    ax.axvline(lam_plus, color="#C44E52", ls="--", lw=1, label=f"noise ceiling λ+={lam_plus:.2f}")
    for o in outliers:
        ax.axvline(o, color="#55A868", lw=1, alpha=0.7)
    ax.set_xlim(0, min(float(np.max(vals)) * 1.05, lam_plus * 3))
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density")
    ax.set_title(title or f"Marchenko-Pastur spectrum — {len(outliers)} factors above the noise floor")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote MP diagnostic plot -> %s (%d signal eigenvalues)", out_path, len(outliers))
    return out_path
