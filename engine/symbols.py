"""Phase 2, step 1 — resolve a canonical instrument to its Yahoo Finance symbol.

Our ``canonical_id`` is a bare ticker slug (``VUSA``, ``SAP``, ``BAE``); Yahoo needs
an exchange-suffixed symbol (``VUSA.L``, ``SAP.DE``, ``BA.L``). Resolution is:

  1. crypto_spot map     — coin exposures use the underlying spot (BTC-USD), not the ETF.
  2. hand overrides      — config/symbol_overrides.yaml, the auditable exceptions.
  3. auto rule           — ticker + a suffix derived from the listing country/exchange
                           (from instrument_metadata.listing_country, itself ISIN-derived
                           or the exchange prefix stripped in Phase 1).

US lines (and US-listed ADRs, whose ISIN is US) get no suffix — the bare ticker works.
Anything the rule can't place is REPORTED (never silently dropped); a coverage probe
turns misses into overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("cov.engine.symbols")

# Country/exchange code -> Yahoo suffix. Keys are ISO-2 (from ISIN) OR the exchange
# code Phase 1 stripped off a colon-prefixed ticker (ASX:, HKG: ...). US/unknown -> "".
SUFFIX_BY_LOCALE: dict[str, str] = {
    # ISO-2 country (from ISIN prefix)
    "US": "", "NL": ".AS", "FR": ".PA", "DE": ".DE", "CH": ".SW", "GB": ".L",
    "CA": ".TO", "IT": ".MI", "IE": ".L", "IL": ".TA", "NO": ".OL", "BE": ".BR",
    "LU": "", "KY": "", "JE": ".L", "GG": ".L", "AN": "",
    # exchange codes (blank-ISIN rows, from the stripped colon prefix)
    "ASX": ".AX", "HKG": ".HK", "CVE": ".V", "CNSX": ".CN", "TSE": ".TO",
    "ETR": ".DE", "FRA": ".DE", "LON": ".L", "EPA": ".PA", "BIT": ".MI", "SWX": ".SW",
}


def load_symbol_overrides(path: str | Path) -> dict:
    """Load the hand-maintained ``canonical_id -> {symbol, currency?}`` override map."""
    path = Path(path)
    if not path.exists():
        logger.warning("No symbol_overrides at %s — auto-rule only", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("symbols", data) or {}


def _auto_symbol(ticker: str, locale: str | None) -> str:
    # A ticker that already carries a dot is exchange-qualified (ENEL.MI, UMI.BR) —
    # use it verbatim; appending another suffix would double it.
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
    """Return one row per canonical: ``canonical_id, ticker, yahoo_symbol, kind, source``.

    ``kind`` is ``crypto_spot`` | ``override`` | ``auto``; ``source`` records the locale
    used, for the coverage report.
    """
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
            if not sym:  # symbol: null -> explicitly unmapped, never faked
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
