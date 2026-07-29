"""Phase 1, step 3 — net across custodians and build the clean netted book.

Given rows already tagged with a ``canonical_id`` (see :mod:`resolution.entity`),
collapse them to one line per canonical instrument: sum money, sum shares only when
share-equivalence holds, retain the contributing ``raw_id`` list for audit, flag
funds for later look-through, and compute portfolio weights.

Because netting only *sums* rows, the total market value is invariant — that is the
Phase 1 reconciliation test (:func:`reconcile`).
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger("cov.resolution.netting")

# Final netted-book column order (the Phase 1 deliverable contract).
BOOK_COLUMNS = [
    "canonical_id", "ticker", "isins", "name", "quantity",
    "cost_basis_usd", "market_value_usd", "weight", "sector_label",
    "contributing_raw_ids", "is_fund", "instrument_type", "resolution_notes",
]


def _fund_flag(names: pd.Series, fees: pd.Series, patterns: list[str]) -> bool:
    """A canonical is a fund/look-through candidate if any line has a fee or a
    fund-like name (``ETF``/``Trust``/``Fund``/``ETC``)."""
    if fees.notna().any():
        return True
    pat = re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)
    return bool(names.dropna().map(lambda n: bool(pat.search(str(n)))).any())


def _mode_or_first(s: pd.Series):
    """Most common non-null value, falling back to the first."""
    s = s.dropna()
    if s.empty:
        return None
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


def net_book(df: pd.DataFrame, fund_patterns: list[str], dq_notes: dict) -> pd.DataFrame:
    """Collapse resolved rows to one line per canonical instrument.

    ``dq_notes`` maps ``raw_id -> note`` (hand-flagged issues). Generic flags
    (``$0`` market value, blank ISIN, mixed sector, non-additive shares) are added
    here so nothing is silently smoothed over.
    """
    dq_notes = {int(k): v for k, v in (dq_notes or {}).items()}
    records = []
    for cid, g in df.groupby("canonical_id", sort=False):
        shares_additive = bool(g["shares_additive"].all())
        raw_ids = sorted(int(x) for x in g["raw_id"])
        isins = sorted({i for i in g["isin"].dropna().unique()})
        sectors = sorted(g["sector_label"].dropna().unique())

        notes: list[str] = []
        for n in g["resolution_note"].unique():
            if isinstance(n, str) and n.strip():
                notes.append(n.strip())
        if not shares_additive:
            notes.append("shares NOT additive across wrappers (MV/cost summed, quantity=NA)")
        if (g["market_value_usd"] == 0).any():
            notes.append("contains a $0 market-value line (stale/needs-price)")
        if g["isin"].isna().any():
            notes.append("blank ISIN on >=1 line — manual entity-resolution gap")
        if len(sectors) > 1:
            notes.append(f"mixed sector_label across lines: {sectors}")
        for rid in raw_ids:
            if rid in dq_notes:
                notes.append(f"raw_id {rid}: {dq_notes[rid]}")

        records.append({
            "canonical_id": cid,
            "ticker": _mode_or_first(g["norm_ticker"]),
            "isins": ";".join(isins) if isins else None,
            "name": _mode_or_first(g["canonical_name"]),
            "quantity": int(g["shares"].sum()) if shares_additive else pd.NA,
            "cost_basis_usd": int(g["cost_basis_usd"].sum()),
            "market_value_usd": int(g["market_value_usd"].sum()),
            "sector_label": _mode_or_first(g["sector_label"]),
            "contributing_raw_ids": ",".join(str(r) for r in raw_ids),
            "is_fund": _fund_flag(g["canonical_name"], g["etf_fee_pct"], fund_patterns),
            "instrument_type": _mode_or_first(g["instrument_type"]),
            "resolution_notes": " | ".join(dict.fromkeys(notes)),  # de-dup, keep order
        })

    book = pd.DataFrame.from_records(records)
    total_mv = int(book["market_value_usd"].sum())
    book["weight"] = book["market_value_usd"] / total_mv if total_mv else 0.0
    book = book.sort_values("market_value_usd", ascending=False).reset_index(drop=True)
    book["quantity"] = book["quantity"].astype("Int64")
    return book[BOOK_COLUMNS]


def reconcile(book: pd.DataFrame, raw_tot: dict, target_mv: int | None, tol: float) -> dict:
    """The Phase 1 sign-off check. Returns a result dict; logs any delta.

    Always verifies the *invariant*: netted MV/cost == parsed raw totals (no rows
    lost or duplicated). If a custodian ``target_mv`` is supplied, also reconciles
    against it to the cent.
    """
    net_mv = int(book["market_value_usd"].sum())
    net_cb = int(book["cost_basis_usd"].sum())
    result = {
        "netted_market_value_usd": net_mv,
        "netted_cost_basis_usd": net_cb,
        "raw_market_value_usd": raw_tot["market_value_usd"],
        "raw_cost_basis_usd": raw_tot["cost_basis_usd"],
        "invariant_ok": abs(net_mv - raw_tot["market_value_usd"]) <= tol
        and abs(net_cb - raw_tot["cost_basis_usd"]) <= tol,
        "instruments": len(book),
    }
    if not result["invariant_ok"]:
        logger.error(
            "RECONCILIATION INVARIANT FAILED: netted MV %s vs raw %s (netting must only sum)",
            net_mv, raw_tot["market_value_usd"],
        )
    else:
        logger.info("Reconciliation invariant OK: netted MV == raw MV == $%s across %d instruments",
                    f"{net_mv:,}", len(book))

    if target_mv is not None:
        delta = net_mv - int(target_mv)
        result["statement_target_usd"] = int(target_mv)
        result["statement_delta_usd"] = delta
        result["statement_ok"] = abs(delta) <= tol
        (logger.info if result["statement_ok"] else logger.warning)(
            "Custodian-statement reconcile: netted %s vs statement %s (delta %s)",
            f"{net_mv:,}", f"{int(target_mv):,}", f"{delta:,}",
        )
    return result


def build_reference_tables(df: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive the DuckDB reference tables (entity_map, instrument_metadata).

    One row per canonical instrument, matching the schemas in ``cov/storage.py``.
    """
    entity_rows, meta_rows = [], []
    book_by_id = book.set_index("canonical_id")
    for cid, g in df.groupby("canonical_id", sort=False):
        isins = sorted({i for i in g["isin"].dropna().unique()})
        primary_isin = isins[0] if isins else None
        b = book_by_id.loc[cid]
        note = " | ".join(
            dict.fromkeys(n.strip() for n in g["resolution_note"].unique()
                          if isinstance(n, str) and n.strip())
        )
        if len(isins) > 1:
            note = (note + f" | merged ISINs: {';'.join(isins)}").strip(" |")
        entity_rows.append({
            "instrument_id": cid,
            "canonical_name": b["name"],
            "isin": primary_isin,
            "ticker": b["ticker"],
            "instrument_type": b["instrument_type"],
            "note": note or None,
        })
        meta_rows.append({
            "instrument_id": cid,
            "sector_label": b["sector_label"],
            "gics_sector": None,                       # filled in Phase 6
            "listing_country": primary_isin[:2] if primary_isin else _mode_first_exchange(g),
            "currency": "USD",                         # base currency; all lines already USD
        })
    return pd.DataFrame(entity_rows), pd.DataFrame(meta_rows)


def _mode_first_exchange(g: pd.DataFrame):
    ex = g["exchange"].dropna()
    return ex.mode().iloc[0] if not ex.empty else None
