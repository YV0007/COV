"""Engine layer — price history, returns, covariance / correlation estimation.

Home of Phase 2 (price history & returns) and Phase 4 (the covariance engine:
Ledoit-Wolf shrinkage, Marchenko-Pastur analysis, RIE/clipping, detoning).
Functions here are stubs until those phases are built.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from engine import fetch, quality, returns as ret, symbols as sym

logger = logging.getLogger("cov.engine")


def _cfg_path(config, value: str) -> Path:
    from config.loader import PROJECT_ROOT
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def build_returns_matrix(config, ctx) -> str:
    """Phase 2: daily adjusted close -> USD log-returns matrix + quality report.

    Resolve each canonical instrument to a Yahoo symbol (rule + overrides), fetch
    auto-adjusted daily close via the yfinance fetcher (behind an interface),
    convert every series to USD (FX captured as a shared factor; pence handled),
    compute log returns, run the quality gates (short history / missing / stale /
    outliers), and store the panel append-only. Deliverable: returns matrix
    (dates × instruments) in ``ctx.artifacts['returns']`` + a per-instrument report.
    """
    p = config.prices
    book = ctx.artifacts.get("book")
    if book is None or len(book) == 0:
        book = ctx.storage.get_book()
    if book is None or len(book) == 0:
        return "error: no netted book found (run Phase 1 first)"
    meta = ctx.storage.get_instrument_metadata()

    # 1. symbols -------------------------------------------------------------
    overrides = sym.load_symbol_overrides(_cfg_path(config, p.symbol_overrides_file))
    symbols = sym.resolve_symbols(book, meta, overrides, p.crypto_spot.to_dict())

    # 2. fetch ---------------------------------------------------------------
    start = str(p.start_date)
    end = ctx.started_at.strftime("%Y-%m-%d")
    fetcher = fetch.YFinanceFetcher()
    prices, failures = fetch.fetch_prices(symbols, start, end, fetcher, int(p.max_workers))
    if prices.empty:
        return "error: no prices fetched"

    # 3. FX -> USD -----------------------------------------------------------
    if bool(p.fx.convert_to_base):
        prices, _fx_missing = ret.to_usd(prices, start, end, fetcher)
    else:
        prices["close_usd"] = prices["close"]

    # 4. log returns + matrix ------------------------------------------------
    returns_long = ret.log_returns(prices, "close_usd")
    matrix = ret.returns_matrix(returns_long)

    # 5. quality gates -------------------------------------------------------
    report = quality.quality_report(prices, returns_long, ctx.run_id,
                                    p.quality_gates, int(p.min_history_days))

    # 6. persist -------------------------------------------------------------
    asof = end
    panel = prices.rename(columns={"close": "close_local"}).merge(
        returns_long, on=["date", "canonical_id"], how="left")
    panel["year"] = pd.to_datetime(panel["date"]).dt.year
    ctx.storage.write_raw_prices(prices[["date", "canonical_id", "yahoo_symbol",
                                         "currency", "close"]], asof)
    for year, chunk in panel.groupby("year"):
        ctx.storage.write_prices(chunk.drop(columns="year"), int(year))
    if not report.empty:
        ctx.storage.record_quality_flags(
            report[["run_id", "instrument_id", "missing_pct", "stale",
                    "outlier_count", "history_days", "flagged"]])

    coverage = round(100.0 * (matrix.shape[1] / len(book)), 2) if len(book) else 0.0
    ctx.storage.record_run_metadata(pd.DataFrame([{
        "run_id": ctx.run_id, "run_date": asof, "phase": "returns",
        "est_window": f"{start}..{end}", "coverage_pct": coverage,
    }]))

    # hand off + report ------------------------------------------------------
    ctx.artifacts["returns"] = matrix
    ctx.artifacts["prices_panel"] = panel
    ctx.artifacts["symbols"] = symbols
    ctx.artifacts["quality"] = report
    ctx.artifacts["price_failures"] = failures

    n_inst = matrix.shape[1] if not matrix.empty else 0
    n_days = matrix.shape[0] if not matrix.empty else 0
    n_flag = int(report["flagged"].sum()) if not report.empty else 0
    unmapped = [f["canonical_id"] for f in failures]
    if unmapped:
        logger.warning("No price for %d instrument(s): %s", len(unmapped), unmapped)
    logger.info("Returns matrix: %d days × %d instruments (%d flagged, %d no-data)",
                n_days, n_inst, n_flag, len(failures))
    return (f"ok: {n_days} days × {n_inst} instruments, "
            f"{n_flag} flagged, {len(failures)} no-data")


def _clean_and_detone(corr_base, delta, n_factors, n, t, window_label, cov_cfg, ctx, run_date):
    """Clean (RIE/clip) then detone one already-chosen base correlation matrix, store both
    results, and return its metadata. ``corr_base`` is the estimator's output for this
    speed (Ledoit-Wolf for full history; the EWMA matrix for EWMA); ``delta`` is the LW
    shrinkage intensity (None where LW does not apply); ``n_factors`` is the MP count from
    this speed's raw spectrum."""
    from engine import covariance as cvx
    _, lam_plus, q = cvx.mp_edges(n, t)
    cleaned = cvx.clean(corr_base, n, t, method=str(cov_cfg.clean))
    detoned, market_share = (cvx.detone(cleaned) if bool(cov_cfg.detone)
                             else (cleaned, float("nan")))
    meta_clean = cvx.matrix_metadata(cleaned)
    meta_det = cvx.matrix_metadata(detoned)
    for label, mat in (("cleaned", cleaned), ("detoned", detoned)):
        ctx.storage.write_snapshot("covariance", run_date,
                                   mat.reset_index(names="instrument"),
                                   name=f"corr_{window_label}_{label}")
    return {
        "window": window_label, "n": n, "t": t, "q": round(q, 4),
        "mp_lambda_plus": round(lam_plus, 4), "n_factors": n_factors,
        "lw_delta": round(delta, 4) if delta is not None else None,
        "market_var_share": round(market_share, 4),
        "effective_rank_cleaned": meta_clean["effective_rank"],
        "effective_rank_detoned": meta_det["effective_rank"],
        "condition_number_cleaned": meta_clean["condition_number"],
    }


def build_covariance(config, ctx) -> str:
    """Phase 4: the statistical core — a trustworthy, denoised correlation map.

    sample correlation -> Marchenko-Pastur noise band -> Ledoit-Wolf shrinkage ->
    RIE/clip eigenvalue cleaning -> detoning (remove the market mode), computed at two
    speeds (full history + EWMA λ=0.94) plus a rolling diagnostics time-series and a
    Spearman robustness cross-check. Every matrix carries metadata (window, estimator,
    condition number, effective rank, coverage). Deliverable: clean matrices + the MP
    diagnostic plot. The matrix is never inverted — this is measurement, not construction.
    """
    from engine import covariance as cvx, windows as win
    cov_cfg = config.covariance

    returns = ctx.artifacts.get("returns")
    if returns is None or getattr(returns, "empty", True):
        returns = ctx.storage.get_returns()
    if returns is None or returns.empty:
        return "error: no returns matrix (run Phase 2 first)"

    # 1. complete estimation block -----------------------------------------
    block = cvx.prepare_block(returns, min_coverage=0.9)
    if block.empty or block.shape[1] < 2:
        return "error: returns block too small after coverage filter"
    n, t = block.shape[1], block.shape[0]
    coverage = round(100.0 * n / returns.shape[1], 2)
    run_date = ctx.started_at.strftime("%Y-%m-%d")
    out_dir = _cfg_path(config, config.paths.reports_out)

    # 2. full-history speed (the structural estimate) ----------------------
    pearson = block.corr()          # the blurry photo (stage 1)
    _lm, lam_plus, q = cvx.mp_edges(n, t)
    raw_vals, _ = cvx.eigidecompose(pearson)          # raw spectrum for MP diagnostic
    n_factors_full = cvx.count_factors(raw_vals, lam_plus)
    # estimator: Ledoit-Wolf shrinks the full-history returns (records δ); else raw sample
    if str(cov_cfg.estimator) == "ledoit_wolf":
        base_full, delta = cvx.ledoit_wolf_corr(block)
    else:
        base_full, delta = pearson, None
    full_meta = _clean_and_detone(base_full, delta, n_factors_full, n, t, "full", cov_cfg, ctx, run_date)

    # MP diagnostic plot on the raw spectrum
    plot_path = win.plot_mp_spectrum(
        raw_vals, q, Path(out_dir) / f"mp_spectrum_{ctx.run_id}.png",
        title=f"MP spectrum — {n} instruments × {t} days — {n_factors_full} factors")

    # 3. EWMA speed (the dynamic estimate) — cleaned directly, LW does not apply ----
    #    EWMA's *effective* sample size is ~(1+λ)/(1-λ), not the full T, so it gets its own
    #    (much wider) MP band and heavier cleaning — an honest q ≈ 5 noise regime.
    lam = float(cov_cfg.windows.ewma_lambda)
    t_eff = (1.0 + lam) / (1.0 - lam)
    ewma_corr = win.ewma_correlation(block, lam=lam)
    ewma_vals, _ = cvx.eigidecompose(ewma_corr)
    _, lam_plus_e, _ = cvx.mp_edges(n, t_eff)
    n_factors_ewma = cvx.count_factors(ewma_vals, lam_plus_e)
    ewma_meta = _clean_and_detone(ewma_corr, None, n_factors_ewma, n, t_eff, "ewma", cov_cfg, ctx, run_date)

    # 4. rolling diagnostics time-series -----------------------------------
    roll = win.rolling_diagnostics(block, window=int(cov_cfg.windows.rolling_days), step=21)
    if not roll.empty:
        ctx.storage.write_snapshot("covariance", run_date, roll, name="rolling_diagnostics")

    # 5. Spearman robustness cross-check -----------------------------------
    spear_flags, spear_stats = win.spearman_agreement(block, pearson)
    if bool(cov_cfg.robustness.spearman) and not spear_flags.empty:
        ctx.storage.write_snapshot("covariance", run_date, spear_flags, name="spearman_divergent_pairs")

    # 6. metadata (honesty labels) -----------------------------------------
    diag = pd.DataFrame([full_meta, ewma_meta])
    diag["run_id"] = ctx.run_id
    diag["coverage_pct"] = coverage
    ctx.storage.write_snapshot("covariance", run_date, diag, name="diagnostics")
    for m in (full_meta, ewma_meta):
        ctx.storage.record_run_metadata(pd.DataFrame([{
            "run_id": ctx.run_id, "run_date": run_date, "phase": "covariance",
            "est_window": m["window"], "estimator": str(cov_cfg.estimator),
            "condition_number": m["condition_number_cleaned"],
            "effective_rank": m["effective_rank_detoned"], "coverage_pct": coverage,
        }]))

    # hand off to Phase 5
    ctx.artifacts["covariance"] = {"full": full_meta, "ewma": ewma_meta}
    ctx.artifacts["spearman_stats"] = spear_stats
    ctx.artifacts["mp_plot"] = str(plot_path)

    logger.info("Covariance: %d×%d block, %d factors above MP, market %.0f%% of variance, "
                "eff.rank %.1f->%.1f (detoned), δ=%s",
                n, t, full_meta["n_factors"], 100 * full_meta["market_var_share"],
                full_meta["effective_rank_cleaned"], full_meta["effective_rank_detoned"],
                full_meta["lw_delta"])
    return (f"ok: {n}×{t}, {full_meta['n_factors']} factors above noise, "
            f"market {100*full_meta['market_var_share']:.0f}% of variance, "
            f"effective bets {full_meta['effective_rank_detoned']:.1f}, "
            f"{spear_stats['n_flagged_pairs']} spearman-divergent pairs")
