# Phase 1, step 1 — ingest & normalize the raw custodian sheet

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("cov.resolution.ingest")

RAW_COLUMNS = [
    "raw_id", "custodian", "sector_label", "ticker", "isin",
    "name", "comment", "etf_fee", "shares", "cost_basis_usd", "market_value_usd",
]


def _split_exchange(ticker: str, prefixes: set[str]) -> tuple[str, str | None]:
    t = (ticker or "").strip()
    if ":" not in t:
        return t, None
    head, _, tail = t.partition(":")
    code = head.strip().upper()
    if code not in prefixes:
        logger.warning("Ticker %r has an unlisted exchange prefix %r — splitting anyway", t, code)
    return tail.strip(), code


def _parse_fee(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        logger.warning("Un-parseable etf_fee %r -> None", value)
        return None


def load_raw_positions(path: str | Path, exchange_prefixes: list[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Custodian sheet not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing expected columns: {sorted(missing)}")

    df["raw_id"] = df["raw_id"].astype(int)
    for col in ("shares", "cost_basis_usd", "market_value_usd"):
        df[col] = pd.to_numeric(df[col].replace("", "0"), errors="raise").astype("int64")

    df["isin"] = df["isin"].str.strip().replace("", pd.NA)
    df["etf_fee_pct"] = df["etf_fee"].map(_parse_fee)

    prefixes = {p.upper() for p in exchange_prefixes}
    split = df["ticker"].map(lambda t: _split_exchange(t, prefixes))
    df["norm_ticker"] = [s[0].upper() for s in split]
    df["exchange"] = [s[1] for s in split]

    logger.info(
        "Ingested %d raw positions from %s (%d custodians, %d blank-ISIN)",
        len(df), path.name, df["custodian"].nunique(), int(df["isin"].isna().sum()),
    )
    return df


def raw_totals(df: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": len(df),
        "market_value_usd": int(df["market_value_usd"].sum()),
        "cost_basis_usd": int(df["cost_basis_usd"].sum()),
    }
