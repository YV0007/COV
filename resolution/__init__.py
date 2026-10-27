# Resolution layer — position -> instrument, entity map, netting, FX, ETF look-through

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from resolution import entity, holdings_parse, holdings_sources, ingest, lookthrough, netting

logger = logging.getLogger("cov.resolution")


def _cfg_path(config, value: str) -> Path:
    from config.loader import PROJECT_ROOT
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def resolve_and_net_positions(config, ctx) -> str:
    ing = config.ingestion
    input_file = _cfg_path(config, ing.input_file)
    overrides = entity.load_overrides(_cfg_path(config, ing.overrides_file))

    raw = ingest.load_raw_positions(input_file, ing.exchange_prefixes)
    raw_tot = ingest.raw_totals(raw)
    logger.info("Raw baseline: %d rows, MV $%s, cost $%s",
                raw_tot["rows"], f"{raw_tot['market_value_usd']:,}",
                f"{raw_tot['cost_basis_usd']:,}")

    resolved = entity.resolve(raw, overrides)

    dq_notes = overrides.get("data_quality_notes", {})
    book = netting.net_book(resolved, list(ing.fund_name_patterns), dq_notes)

    recon = config.ingestion.reconciliation
    target = recon.get("statement_total_market_value_usd")
    result = netting.reconcile(book, raw_tot, target, float(recon.tolerance_usd))
    if not result["invariant_ok"]:
        return "error: reconciliation invariant failed (see log)"

    entity_map, meta = netting.build_reference_tables(resolved, book)
    run_date = ctx.started_at.strftime("%Y-%m-%d")
    book_path = ctx.storage.write_snapshot("book", run_date, book, name="netted_book")
    ctx.storage.upsert_reference("entity_map", entity_map)
    ctx.storage.upsert_reference("instrument_metadata", meta)

    ctx.artifacts["book"] = book
    ctx.artifacts["book_path"] = book_path
    ctx.artifacts["reconciliation"] = result

    n_funds = int(book["is_fund"].sum())
    logger.info("Netted book: %d instruments (%d funds), weights sum %.6f -> %s",
                len(book), n_funds, float(book["weight"].sum()), book_path)
    return (f"ok: {len(book)} instruments, MV ${result['netted_market_value_usd']:,} "
            f"reconciled to the cent")


def _load_constituent_overrides(path: Path) -> dict:
    import yaml
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("overrides", data) or {}


def etf_lookthrough(config, ctx) -> str:
    lt = config.lookthrough
    book = ctx.artifacts.get("book")
    if book is None or len(book) == 0:
        book = ctx.storage.get_book()
    if book is None or len(book) == 0:
        return "error: no netted book found (run Phase 1 first)"

    classification = lookthrough.load_classification(_cfg_path(config, lt.classification_file))
    treatments, unclassified = lookthrough.classify_funds(book, classification)
    equity_funds = [cid for cid, t in treatments.items() if t == "equity"]

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

    overrides = _load_constituent_overrides(_cfg_path(config, lt.overrides_file))
    exposures, fund_recon = lookthrough.build_effective_exposures(
        book, holdings, treatments, overrides,
        recon_tol=float(lt.reconciliation.weight_sum_tolerance), proxy_map=proxy_map)

    total = lookthrough.reconcile_total(exposures, book,
                                        tol=float(lt.reconciliation.total_tolerance))
    if not total["reconciled"]:
        return "error: look-through leaked weight (see log)"
    report = lookthrough.concentration_report(exposures, top_n=int(lt.report_top_n))

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
