"""Phase 5 — structure detection (the core-insight maths).

Turns Phase 4's clean, market-removed correlation matrix into the human-readable answer
to the project's central question: how many real groups does the book contain, which
holdings secretly co-move, and where does behaviour disagree with the declared sectors.

  PCA (interpret)   — read the detoned eigenvectors: which holdings drive each factor.
  clustering        — correlation distance sqrt(2(1-rho)) + Ward linkage -> a tree.
  sector confusion  — data clusters vs the ~20 declared sectors: agreement + the mislabels.
  absorption ratio  — top-N eigenvalue share over time (systemic-risk / fragility dial).

Nothing here reduces or projects the data away: the full N×N matrix is used so every
holding keeps its identity for clustering and attribution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("cov.analytics.structure")


# --- PCA interpretation (read Phase 4's eigenvectors) -------------------------
def pca_factors(corr: pd.DataFrame, sectors: dict, n: int = 12, top_k: int = 7) -> pd.DataFrame:
    """For the top ``n`` factors of ``corr`` (detoned), list the heaviest-loading holdings
    and their dominant declared sector — i.e. name each 'dial' by what drives it."""
    from engine.covariance import eigidecompose
    vals, vecs = eigidecompose(corr)
    total = float(vals.sum())
    keep = min(n, int((vals > 0).sum()))
    rows = []
    for k in range(keep):
        load = pd.Series(vecs[:, k], index=corr.index)
        top = load.reindex(load.abs().sort_values(ascending=False).index).head(top_k)
        secs = [sectors.get(i, "?") for i in top.index]
        dom = pd.Series(secs).mode().iloc[0] if secs else "?"
        rows.append({
            "factor": k + 1,
            "variance_share": round(float(vals[k] / total), 4) if total else 0.0,
            "dominant_sector": dom,
            "drivers": ";".join(map(str, top.index)),
        })
    return pd.DataFrame(rows)


# --- correlation distance + hierarchical clustering ---------------------------
def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """The metric d = sqrt(2(1-rho)): rho=1 -> 0, rho=0 -> sqrt2, rho=-1 -> 2."""
    d = np.sqrt(np.clip(2.0 * (1.0 - corr.values), 0.0, None))
    np.fill_diagonal(d, 0.0)
    d = 0.5 * (d + d.T)                                    # enforce symmetry
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def hierarchical_clusters(corr: pd.DataFrame, method: str = "ward", k: int = 8):
    """Ward-linkage agglomerative clustering on the correlation distance.

    Returns (linkage matrix Z, cluster-label Series). ``k`` clusters are cut from the tree
    with ``fcluster`` (maxclust). Z drives the dendrogram plot.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    condensed = squareform(correlation_distance(corr).values, checks=False)
    z = linkage(condensed, method=method)
    labels = fcluster(z, t=k, criterion="maxclust")
    return z, pd.Series(labels, index=corr.index, name="cluster")


# --- data clusters vs declared sectors ---------------------------------------
def sector_confusion(clusters: pd.Series, sectors: dict):
    """Cross-tab data-driven clusters against declared sectors; agreement + the mislabels.

    Returns (confusion DataFrame, stats dict, per-holding frame flagging holdings that do
    NOT sit in the majority cluster of their declared sector = behaviourally divergent).
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    df = pd.DataFrame({"cluster": clusters,
                       "sector": pd.Series(sectors).reindex(clusters.index)}).dropna()
    conf = pd.crosstab(df["sector"], df["cluster"])
    ari = adjusted_rand_score(df["sector"], df["cluster"])
    nmi = normalized_mutual_info_score(df["sector"], df["cluster"])
    majority = df.groupby("sector")["cluster"].agg(lambda s: s.mode().iloc[0])
    df["sector_majority_cluster"] = df["sector"].map(majority)
    df["behaviorally_divergent"] = df["cluster"] != df["sector_majority_cluster"]
    stats = {"adjusted_rand": round(float(ari), 4), "nmi": round(float(nmi), 4),
             "n_divergent": int(df["behaviorally_divergent"].sum()), "n": len(df),
             "n_clusters": int(clusters.nunique()), "n_sectors": int(df["sector"].nunique())}
    return conf, stats, df.reset_index(names="instrument")


# --- absorption ratio (systemic-risk dial) -----------------------------------
def absorption_ratio(eigenvalues: np.ndarray, top_n: int = 5) -> float:
    """Fraction of total variance captured by the top-N eigenvectors (Kritzman 2010).
    High = tightly coupled / fragile; low = variance spread across many factors."""
    v = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    s = v.sum()
    return float(v[:top_n].sum() / s) if s else float("nan")


def rolling_absorption(block: pd.DataFrame, top_n: int = 5,
                       window: int = 252, step: int = 21) -> pd.DataFrame:
    """Absorption ratio over rolling windows — the systemic-fragility trend."""
    idx = block.index
    rows = []
    for end in range(window, len(block) + 1, step):
        sub = block.iloc[end - window:end]
        sub = sub.loc[:, sub.std(axis=0) > 0]
        if sub.shape[1] < top_n + 1:
            continue
        c = np.nan_to_num(sub.corr().values, nan=0.0)
        np.fill_diagonal(c, 1.0)
        rows.append({"date": idx[end - 1],
                     "absorption_ratio": round(absorption_ratio(np.linalg.eigvalsh(c), top_n), 4),
                     "n_instruments": sub.shape[1]})
    return pd.DataFrame(rows)


# --- plots -------------------------------------------------------------------
def plot_dendrogram(z, labels, k: int, out_path: Path, title: str = "") -> Path:
    """Save the clustering dendrogram, coloured at the k-cluster cut."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    # colour threshold = the merge height that yields k clusters
    heights = np.sort(z[:, 2])
    thresh = heights[-(k - 1)] if k > 1 and k <= len(heights) else None
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(z, labels=list(labels), color_threshold=thresh, no_labels=len(labels) > 40, ax=ax)
    if thresh:
        ax.axhline(thresh, color="grey", ls="--", lw=0.8)
    ax.set_title(title or f"Holdings clustered by co-movement — cut into {k} groups")
    ax.set_ylabel("correlation distance  sqrt(2(1-rho))")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_absorption(series: pd.DataFrame, out_path: Path, title: str = "") -> Path:
    """Save the absorption-ratio-over-time chart (systemic fragility trend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pd.to_datetime(series["date"]), series["absorption_ratio"], color="#C44E52", lw=1.6)
    ax.fill_between(pd.to_datetime(series["date"]), series["absorption_ratio"], color="#C44E52", alpha=0.15)
    ax.set_title(title or "Absorption ratio over time — systemic coupling (higher = more fragile)")
    ax.set_ylabel("top-5 eigenvalue share")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
