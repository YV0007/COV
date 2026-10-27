# Phase 3, step 2 — normalize an issuer holdings CSV into one canonical schema

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger("cov.resolution.holdings_parse")

HOLDING_COLUMNS = [
    "fund_id", "constituent_name", "ticker", "isin", "asset_class", "weight_in_fund",
]

FIELD_ALIASES: dict[str, list[str]] = {
    "constituent_name": ["name", "security name", "holding name", "description", "issuer name"],
    "ticker": ["ticker", "holding ticker", "issuer ticker", "symbol", "local ticker"],
    "isin": ["isin"],
    "asset_class": ["asset class", "class of shares", "security type", "type", "asset type"],
    "weight_pct": ["weight (%)", "weight", "% of net assets", "portfolio weight",
                   "% weight", "weighting", "market value weight",
                   "percent of fund", "% of fund"],
}

_ASSET_CLASS_RULES: list[tuple[str, str]] = [
    ("money market", "cash"),
    ("cash", "cash"),
    ("deriv", "derivative"),
    ("future", "derivative"),
    ("forward", "derivative"),
    ("swap", "derivative"),
    ("option", "derivative"),
    ("bond", "bond"),
    ("treasury", "bond"),
    ("fixed income", "bond"),
    ("note", "bond"),
    ("commodity", "commodity"),
    ("equity", "equity"),
    ("stock", "equity"),
    ("common", "equity"),
    ("share", "equity"),
]

_CASH_NAME_RE = re.compile(
    r"\b(cash|usd|eur|gbp|collateral|margin|net other assets|fx|receivable|payable)\b",
    re.IGNORECASE,
)


def _find_column(cols: list[str], aliases: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in cols}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def _normalize_asset_class(raw: object, name: str) -> str:
    text = str(raw).lower() if raw is not None and not pd.isna(raw) else ""
    for needle, label in _ASSET_CLASS_RULES:
        if needle in text:
            return label
    if _CASH_NAME_RE.search(str(name or "")):
        return "cash"
    return "equity"


def _parse_weight(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    vals = pd.to_numeric(s, errors="coerce")
    total = vals.abs().sum(skipna=True)
    if total is not None and total > 2.0:
        vals = vals / 100.0
    return vals


def normalize_holdings(raw: pd.DataFrame, fund_id: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=HOLDING_COLUMNS)

    cols = list(raw.columns)
    picked = {field: _find_column(cols, aliases) for field, aliases in FIELD_ALIASES.items()}
    if picked["weight_pct"] is None:
        raise ValueError(
            f"{fund_id}: no weight column found in holdings CSV (looked for "
            f"{FIELD_ALIASES['weight_pct']}); columns were {cols}"
        )

    out = pd.DataFrame(index=range(len(raw)))
    out["fund_id"] = fund_id
    out["constituent_name"] = (
        raw[picked["constituent_name"]].astype(str).str.strip()
        if picked["constituent_name"] else None
    )
    out["ticker"] = (
        raw[picked["ticker"]].astype(str).str.strip().replace({"nan": None, "": None})
        if picked["ticker"] else None
    )
    out["isin"] = (
        raw[picked["isin"]].astype(str).str.strip().replace({"nan": None, "": None})
        if picked["isin"] else None
    )
    raw_ac = raw[picked["asset_class"]] if picked["asset_class"] else pd.Series([None] * len(raw))
    names = out["constituent_name"] if "constituent_name" in out else pd.Series([None] * len(raw))
    out["asset_class"] = [
        _normalize_asset_class(ac, nm) for ac, nm in zip(raw_ac, names)
    ]
    out["weight_in_fund"] = _parse_weight(raw[picked["weight_pct"]]).to_numpy()

    out = out[out["weight_in_fund"].notna()].reset_index(drop=True)
    missing = [f for f in ("ticker", "isin", "asset_class") if picked[f] is None]
    if missing:
        logger.info("%s: holdings CSV missing %s — resolver will lean on remaining ids",
                    fund_id, missing)
    return out[HOLDING_COLUMNS]
