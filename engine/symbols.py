# Phase 2, step 1 — resolve a canonical instrument to its Yahoo Finance symbol

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("cov.engine.symbols")

SUFFIX_BY_LOCALE: dict[str, str] = {
    "US": "", "NL": ".AS", "FR": ".PA", "DE": ".DE", "CH": ".SW", "GB": ".L",
    "CA": ".TO", "IT": ".MI", "IE": ".L", "IL": ".TA", "NO": ".OL", "BE": ".BR",
    "LU": "", "KY": "", "JE": ".L", "GG": ".L", "AN": "",
    "ASX": ".AX", "HKG": ".HK", "CVE": ".V", "CNSX": ".CN", "TSE": ".TO",
    "ETR": ".DE", "FRA": ".DE", "LON": ".L", "EPA": ".PA", "BIT": ".MI", "SWX": ".SW",
}


def load_symbol_overrides(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        logger.warning("No symbol_overrides at %s — auto-rule only", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("symbols", data) or {}


def _auto_symbol(ticker: str, locale: str | None) -> str:
    if "." in ticker:
        return ticker
    code = locale.upper() if isinstance(locale, str) else ""
    return f"{ticker}{SUFFIX_BY_LOCALE.get(code, '')}"


def resolve_symbols(
    book: pd.DataFrame,
    metadata: pd.DataFrame,
    overrides: dict,
    crypto_spot: dict,
) -> pd.DataFrame:
    locale_by_id = (
        metadata.set_index("instrument_id")["listing_country"].to_dict()
        if not metadata.empty else {}
    )
    crypto_spot = {k: v for k, v in (crypto_spot or {}).items()}
    rows = []
    for r in book.itertuples(index=False):
        cid = r.canonical_id
        if cid in crypto_spot:
            rows.append((cid, r.ticker, crypto_spot[cid], "crypto_spot", "coin-spot"))
        elif cid in overrides:
            ov = overrides[cid]
            sym = ov.get("symbol") if isinstance(ov, dict) else ov
            if not sym:
                rows.append((cid, r.ticker, None, "unmapped", "override"))
            else:
                rows.append((cid, r.ticker, sym, "override", "override"))
        else:
            locale = locale_by_id.get(cid)
            rows.append((cid, r.ticker, _auto_symbol(r.ticker, locale), "auto", str(locale)))
    out = pd.DataFrame(rows, columns=["canonical_id", "ticker", "yahoo_symbol", "kind", "source"])
    logger.info("Resolved %d symbols (%d overrides, %d crypto-spot, %d auto)",
                len(out), (out.kind == "override").sum(),
                (out.kind == "crypto_spot").sum(), (out.kind == "auto").sum())
    return out
