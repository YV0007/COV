"""Resolution layer — position -> instrument, entity map, netting, FX, ETF look-through.

Home of Phase 1 (data model & instrument resolution) and Phase 3 (ETF look-through).
Phase 1 is built (ingest/entity/netting); Phase 3 remains a stub.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from resolution import entity, holdings_parse, holdings_sources, ingest, lookthrough, netting

logger = logging.getLogger("cov.resolution")


def _cfg_path(config, value: str) -> Path:
    """Resolve a config-relative path against the project root."""
    from config.loader import PROJECT_ROOT
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def resolve_and_net_positions(config, ctx) -> str:
    """Phase 1: ingest custodian sheet -> canonical instruments -> netted book.

    Ingest & normalize the raw rows, resolve each to a canonical instrument via the
    hand-maintained entity map (ASML dual ISIN, IBIT+CBTC -> BTC, BA/BAE split,
    BMW share classes), net across custodians, flag funds, compute weights, and
    write one clean netted book (Parquet snapshot) plus the DuckDB reference
    tables. Reconciles the netted total against the parsed baseline to the cent.

    FX is a no-op at Phase 1: the money columns are already USD (the base
    currency). The conversion hook stays in the pipeline for future non-USD sheets.
    """
    ing = config.ingestion
    input_file = _cfg_path(config, ing.input_file)
    overrides = entity.load_overrides(_cfg_path(config, ing.overrides_file))

    # 1. ingest & normalize --------------------------------------------------
    raw = ingest.load_raw_positions(input_file, ing.exchange_prefixes)
    raw_tot = ingest.raw_totals(raw)
    logger.info("Raw baseline: %d rows, MV $%s, cost $%s",
                raw_tot["rows"], f"{raw_tot['market_value_usd']:,}",
                f"{raw_tot['cost_basis_usd']:,}")

    # 2. entity resolution ---------------------------------------------------
    resolved = entity.resolve(raw, overrides)

    # 3. net & build book ----------------------------------------------------
    dq_notes = overrides.get("data_quality_notes", {})
    book = netting.net_book(resolved, list(ing.fund_name_patterns), dq_notes)

    # 4. reconcile (the Phase 1 sign-off test) -------------------------------
    recon = config.ingestion.reconciliation
    target = recon.get("statement_total_market_value_usd")
    result = netting.reconcile(book, raw_tot, target, float(recon.tolerance_usd))
    if not result["invariant_ok"]:
        return "error: reconciliation invariant failed (see log)"

    # 5. persist: netted book snapshot + DuckDB reference tables -------------
    entity_map, meta = netting.build_reference_tables(resolved, book)
    run_date = ctx.started_at.strftime("%Y-%m-%d")
    book_path = ctx.storage.write_snapshot("book", run_date, book, name="netted_book")
    ctx.storage.upsert_reference("entity_map", entity_map)
    ctx.storage.upsert_reference("instrument_metadata", meta)

    # hand off to later phases
    ctx.artifacts["book"] = book
    ctx.artifacts["book_path"] = book_path
    ctx.artifacts["reconciliation"] = result

    n_funds = int(book["is_fund"].sum())
    logger.info("Netted book: %d instruments (%d funds), weights sum %.6f -> %s",
                len(book), n_funds, float(book["weight"].sum()), book_path)
    return (f"ok: {len(book)} instruments, MV ${result['netted_market_value_usd']:,} "
            f"reconciled to the cent")


def _load_constituent_overrides(path: Path) -> dict:
    """Load the identifier->canonical_id constituent override map (empty if absent)."""
    import yaml
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("overrides", data) or {}


def etf_lookthrough(config, ctx) -> str:
    """Phase 3: decompose ETFs into effective single-name exposure.

    Classify each flagged fund (equity/commodity/crypto/not_a_fund), fetch issuer
    holdings CSVs for the equity baskets, resolve every constituent onto the book's
    canonical_id space, compute effective exposure = ETF weight x constituent weight,
    aggregate with direct holdings, and reconcile each fund's weight to the cent.
    Deliverable: the effective-exposure table + "declared vs. true concentration"
    report — the first client-visible result.

    Fetching is source-driven: with an empty ``etf_holdings_sources.yaml`` nothing is
    downloaded and every equity fund is honestly reported as un-looked-through ballast.
    Fill the registry and the look-through lights up fund by fund.
    """
    lt = config.lookthrough
    book = ctx.artifacts.get("book")
    if book is None or len(book) == 0:
        book = ctx.storage.get_book()
    if book is None or len(book) == 0:
        return "error: no netted book found (run Phase 1 first)"

    # 1. classify flagged funds ---------------------------------------------
    classification = lookthrough.load_classification(_cfg_path(config, lt.classification_file))
    treatments, unclassified = lookthrough.classify_funds(book, classification)
    equity_funds = [cid for cid, t in treatments.items() if t == "equity"]

    # 2. fetch + normalize holdings for the equity baskets ------------------
    # A fund may declare `proxy_of: <other_fund>` to reuse an index-equivalent fund's
    # holdings (e.g. a gated S&P 500 tracker <- a liquid S&P 500 ETF). Proxies are resolved
    # after the real downloads, by cloning the source fund's normalized constituents.
    from config.loader import PROJECT_ROOT
    sources = holdings_sources.load_holdings_sources(_cfg_path(config, lt.sources_file))
    proxy_map = {fid: sources[fid]["proxy_of"] for fid in equity_funds
                 if fid in sources and sources[fid].get("proxy_of")}
    direct_ids = [fid for fid in equity_funds if fid not in proxy_map]

    fetcher = holdings_sources.IssuerHoldingsFetcher(
        asof=ctx.started_at.strftime("%Y-%m-%d"), project_root=PROJECT_ROOT)
    raw_by_fund, misses = holdings_sources.fetch_holdings(direct_ids, sources, fetcher)
    normalized: dict[str, pd.DataFrame] = {}
    for fid, raw in raw_by_fund.items():
        try:
            normalized[fid] = holdings_parse.normalize_holdings(raw, fid)
        except ValueError as exc:
            logger.warning("Holdings parse failed for %s: %s", fid, exc)
            misses.append({"fund_id": fid, "reason": "parse_error"})

    # resolve proxies: clone the source fund's constituents under the proxied fund_id
    for fid, src_fid in proxy_map.items():
        base = normalized.get(src_fid)
        if base is None or base.empty:
            misses.append({"fund_id": fid, "reason": f"proxy_source_{src_fid}_unavailable"})
            continue
        clone = base.copy()
        clone["fund_id"] = fid
        normalized[fid] = clone
        logger.info("Proxied %s holdings from %s (index-equivalent)", fid, src_fid)

    frames = list(normalized.values())
    holdings = (pd.concat(frames, ignore_index=True)
                if frames else pd.DataFrame(columns=holdings_parse.HOLDING_COLUMNS))

    # 3. resolve + decompose + aggregate ------------------------------------
    overrides = _load_constituent_overrides(_cfg_path(config, lt.overrides_file))
    exposures, fund_recon = lookthrough.build_effective_exposures(
        book, holdings, treatments, overrides,
        recon_tol=float(lt.reconciliation.weight_sum_tolerance), proxy_map=proxy_map)

    # 4. reconcile (the Phase 3 sign-off invariant) -------------------------
    total = lookthrough.reconcile_total(exposures, book,
                                        tol=float(lt.reconciliation.total_tolerance))
    if not total["reconciled"]:
        return "error: look-through leaked weight (see log)"
    report = lookthrough.concentration_report(exposures, top_n=int(lt.report_top_n))

    # 5. persist ------------------------------------------------------------
    run_date = ctx.started_at.strftime("%Y-%m-%d")
    asof_month = ctx.started_at.strftime("%Y-%m")
    if not holdings.empty:
        ctx.storage.write_holdings_snapshot(holdings, asof_month)
    exp_path = ctx.storage.write_snapshot("exposure", run_date, exposures, name="effective_exposure")
    n_looked = int(fund_recon.get("looked_through", pd.Series(dtype=bool)).sum()) if not fund_recon.empty else 0
    looked_mv = float(book.loc[book["canonical_id"].isin(
        [r["fund_id"] for r in fund_recon.to_dict("records") if r.get("looked_through")]),
        "weight"].sum()) if not fund_recon.empty else 0.0
    ctx.storage.record_run_metadata(pd.DataFrame([{
        "run_id": ctx.run_id, "run_date": run_date, "phase": "lookthrough",
        "est_window": asof_month,
        "coverage_pct": round(100.0 * looked_mv / float(book["weight"].sum()), 2),
    }]))

    # hand off to later phases
    ctx.artifacts["exposures"] = exposures
    ctx.artifacts["fund_recon"] = fund_recon
    ctx.artifacts["concentration"] = report
    ctx.artifacts["lookthrough_misses"] = misses

    m = report["metrics"]
    logger.info("Look-through: %d/%d equity funds decomposed, %d flagged funds classified "
                "(%d unclassified). Effective names %s -> %s. -> %s",
                n_looked, len(equity_funds), len(treatments), len(unclassified),
                m["effective_names_declared"], m["effective_names_true"], exp_path)
    return (f"ok: {n_looked}/{len(equity_funds)} equity funds looked through, "
            f"{len(exposures)} effective exposures, reconciled to the cent")
