# Analytics layer — structure detection, factor overlay, risk attribution, stress

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("cov.analytics")


def _cfg_path(config, value: str) -> Path:
    from config.loader import PROJECT_ROOT
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    m = df.set_index("instrument") if "instrument" in df.columns else df
    ids = [c for c in m.index if c in m.columns]
    return m.loc[ids, ids]


def detect_structure(config, ctx) -> str:
    from analytics import structure as sx
    from engine import covariance as cvx
    scfg = config.structure
    run_date = ctx.started_at.strftime("%Y-%m-%d")
    out_dir = _cfg_path(config, config.paths.reports_out)

    mats = ctx.artifacts.get("cov_matrices", {})
    detoned = mats.get("detoned")
    cleaned = mats.get("cleaned")
    if detoned is None or getattr(detoned, "empty", True):
        detoned = _matrix(ctx.storage.get_covariance("corr_full_detoned"))
    if cleaned is None or getattr(cleaned, "empty", True):
        cleaned = _matrix(ctx.storage.get_covariance("corr_full_cleaned"))
    if detoned is None or detoned.empty:
        return "error: no cleaned covariance matrix found (run Phase 4 first)"

    cov_meta = ctx.artifacts.get("covariance", {}).get("full", {})
    n_factors = cov_meta.get("n_factors")
    if n_factors is None:
        diag = ctx.storage.get_covariance("diagnostics")
        if not diag.empty and "n_factors" in diag.columns:
            row = diag[diag["window"] == "full"] if "window" in diag.columns else diag
            n_factors = int(row["n_factors"].iloc[0]) if len(row) else None
    k = int(np.clip(n_factors or 8, 2, 30))

    sectors = ctx.storage.get_instrument_metadata().set_index("instrument_id")["sector_label"].to_dict()

    factors = sx.pca_factors(detoned, sectors, n=min(k, 12))

    z, clusters = sx.hierarchical_clusters(detoned, method=str(scfg.clustering_linkage), k=k)
    dendro_path = sx.plot_dendrogram(z, clusters.index, k,
                                     Path(out_dir) / f"dendrogram_{ctx.run_id}.png")

    confusion, agree, per_holding = sx.sector_confusion(clusters, sectors)

    top_n = int(scfg.absorption_ratio_top_n)
    ar_static = sx.absorption_ratio(np.linalg.eigvalsh(cleaned.values), top_n) if not cleaned.empty else float("nan")
    returns = ctx.artifacts.get("returns")
    if returns is None or getattr(returns, "empty", True):
        returns = ctx.storage.get_returns()
    block = cvx.prepare_block(returns, min_coverage=0.9)
    ar_series = sx.rolling_absorption(block, top_n=top_n, window=252, step=21) if not block.empty else pd.DataFrame()
    ar_plot = (sx.plot_absorption(ar_series, Path(out_dir) / f"absorption_{ctx.run_id}.png")
               if not ar_series.empty else None)

    ctx.storage.write_snapshot("covariance", run_date, factors, name="factor_table")
    ctx.storage.write_snapshot("covariance", run_date,
                               clusters.rename_axis("instrument").reset_index(), name="cluster_assignments")
    ctx.storage.write_snapshot("covariance", run_date,
                               confusion.reset_index(), name="sector_confusion")
    divergent = per_holding[per_holding["behaviorally_divergent"]]
    if not divergent.empty:
        ctx.storage.write_snapshot("covariance", run_date, divergent, name="mislabeled_holdings")
    if not ar_series.empty:
        ctx.storage.write_snapshot("covariance", run_date, ar_series, name="absorption_series")

    ctx.artifacts["structure"] = {
        "k": k, "factor_table": factors, "clusters": clusters, "confusion": confusion,
        "agreement": agree, "divergent": divergent, "absorption_ratio": round(ar_static, 4),
        "dendrogram": str(dendro_path), "absorption_plot": str(ar_plot) if ar_plot else None,
    }
    logger.info("Structure: %d clusters vs %d declared sectors — ARI=%.2f, NMI=%.2f, "
                "%d behaviourally-divergent holdings; absorption ratio (top-%d)=%.2f",
                agree["n_clusters"], agree["n_sectors"], agree["adjusted_rand"],
                agree["nmi"], agree["n_divergent"], top_n, ar_static)
    return (f"ok: {k} factors/clusters vs {agree['n_sectors']} declared sectors, "
            f"ARI={agree['adjusted_rand']}, {agree['n_divergent']} mislabeled holdings, "
            f"absorption(top-{top_n})={ar_static:.2f}")


def factor_overlay(config, ctx) -> str:
    from analytics import factors as fx
    from engine import covariance as cvx

    returns = ctx.artifacts.get("returns")
    if returns is None or getattr(returns, "empty", True):
        returns = ctx.storage.get_returns()
    if returns is None or returns.empty:
        return "error: no returns matrix (run Phase 2 first)"

    run_date = ctx.started_at.strftime("%Y-%m-%d")
    start = str(returns.index.min())[:10]
    end = str(returns.index.max())[:10]

    fetcher = fx.FredFactorFetcher()
    panel, sources = fx.build_factor_panel(returns, fetcher, start, end)
    if panel.shape[1] == 0:
        return "error: no factors could be built"
    logger.info("Factor panel: %d factors %s", panel.shape[1], sources)

    betas, r2 = fx.regress_holdings(returns, panel)

    book = ctx.artifacts.get("book")
    if book is None or getattr(book, "empty", True):
        book = ctx.storage.get_book()
    weights = book.set_index("canonical_id")["weight"]
    exposure = fx.portfolio_exposure(betas, weights)

    detoned = _matrix(ctx.storage.get_covariance("corr_full_detoned"))
    naming = pd.DataFrame()
    if not detoned.empty:
        block = cvx.prepare_block(returns, min_coverage=0.9)
        pca_ret = fx.pca_factor_returns(detoned, block, k=8)
        naming = fx.name_factors(pca_ret, panel)

    ctx.storage.write_snapshot("covariance", run_date,
                               betas.reset_index(names="instrument"), name="factor_betas")
    ctx.storage.write_snapshot("covariance", run_date, exposure.reset_index(names="factor"),
                               name="portfolio_factor_exposure")
    if not naming.empty:
        ctx.storage.write_snapshot("covariance", run_date, naming, name="pca_factor_names")

    ctx.artifacts["factors"] = {
        "panel_sources": sources, "betas": betas, "r2": r2,
        "exposure": exposure, "naming": naming,
    }
    top = exposure.index[0] if len(exposure) else "?"
    logger.info("Factor overlay: %d factors, mean R²=%.2f, largest net exposure = %s (%.3f)",
                panel.shape[1], float(r2.mean()) if len(r2) else float("nan"),
                top, exposure["net_beta"].iloc[0] if len(exposure) else float("nan"))
    return (f"ok: {panel.shape[1]} factors ({'+'.join(sources)}), mean R²={float(r2.mean()):.2f}, "
            f"top net exposure {top}={exposure['net_beta'].iloc[0]:.3f}")


def risk_attribution_and_stress(config, ctx) -> str:
    from analytics import stress as ss

    cleaned = _matrix(ctx.storage.get_covariance("corr_full_cleaned"))
    if cleaned is None or cleaned.empty:
        return "error: no cleaned covariance matrix (run Phase 4 first)"

    returns = ctx.artifacts.get("returns")
    if returns is None or getattr(returns, "empty", True):
        returns = ctx.storage.get_returns()
    book = ctx.artifacts.get("book")
    if book is None or getattr(book, "empty", True):
        book = ctx.storage.get_book()

    universe = [c for c in cleaned.index if c in returns.columns]
    corr = cleaned.loc[universe, universe]
    vols = returns[universe].std()
    weights = book.set_index("canonical_id")["weight"].reindex(universe).fillna(0.0)
    covered_w = float(weights.sum())
    weights = weights / covered_w if covered_w else weights

    cov = ss.cov_from_corr(corr, vols)
    contrib, sig_daily = ss.risk_contributions(cov, weights)
    ann_vol = sig_daily * ss.ANNUALIZE
    enb = ss.effective_risk_bets(contrib["pct_risk"])

    clus_df = ctx.storage.get_covariance("cluster_assignments")
    cluster_tab = pd.DataFrame()
    if not clus_df.empty and "instrument" in clus_df.columns:
        clusters = clus_df.set_index("instrument")["cluster"]
        cluster_tab = ss.cluster_risk(contrib, clusters)

    episodes = {k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in config.stress.episodes.to_dict().items()}
    all_to = float(config.stress.scenarios.all_correlations_to)
    stab = ss.stress_table(corr, vols, weights, returns, episodes, all_to)

    run_date = ctx.started_at.strftime("%Y-%m-%d")
    ctx.storage.write_snapshot("covariance", run_date,
                               contrib.reset_index(names="instrument"), name="risk_contributions")
    if not cluster_tab.empty:
        ctx.storage.write_snapshot("covariance", run_date,
                                   cluster_tab.reset_index(), name="cluster_risk")
    ctx.storage.write_snapshot("covariance", run_date, stab, name="stress_table")

    ctx.artifacts["risk"] = {
        "annual_vol": round(ann_vol, 4), "effective_risk_bets": round(enb, 2),
        "contributions": contrib, "cluster_risk": cluster_tab, "stress": stab,
        "weight_covered": round(covered_w, 4),
    }
    tail = stab[stab["scenario"].str.startswith("all_corr")]["vs_baseline"]
    logger.info("Risk: annual vol %.1f%%, %.1f effective risk bets; all-corr tail = %.2fx baseline",
                100 * ann_vol, enb, float(tail.iloc[0]) if len(tail) else float("nan"))
    return (f"ok: annual vol {100*ann_vol:.1f}%, {enb:.1f} effective risk bets, "
            f"crisis tail {float(tail.iloc[0]):.2f}x baseline vol"
            if len(tail) else f"ok: annual vol {100*ann_vol:.1f}%, {enb:.1f} effective risk bets")
