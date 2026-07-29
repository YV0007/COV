"""Phase 3, step 2 — normalize an issuer holdings CSV into one canonical schema.

Every issuer (iShares, Vanguard, SSGA/SPDR, Invesco, VanEck, ...) ships a slightly
different CSV, but they overwhelmingly share the same *column names* ("Name",
"Ticker", "ISIN", "Weight (%)", "Asset Class"). So rather than a bespoke parser per
issuer, this normalizes by **column aliases** plus an asset-class vocabulary — the
same config-driven, surfaced-not-silent spirit as the rest of the pipeline. Per-issuer
quirks get pinned down against live samples in the source registry when we wire real
URLs; the generic path already covers the common case.

Output schema (one row per constituent line):

    fund_id, constituent_name, ticker, isin, asset_class, weight_in_fund

``weight_in_fund`` is a FRACTION (0..1), not a percent. ``asset_class`` is normalized
to a small vocabulary so downstream can split investable equity from the cash /
derivative residue **without dropping it** — the residue is what makes each fund
reconcile back to its own weight.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger("cov.resolution.holdings_parse")

HOLDING_COLUMNS = [
    "fund_id", "constituent_name", "ticker", "isin", "asset_class", "weight_in_fund",
]

# Our field -> candidate source column names (matched case-insensitively, first hit
# wins). Extend as new issuer layouts appear.
FIELD_ALIASES: dict[str, list[str]] = {
    "constituent_name": ["name", "security name", "holding name", "description", "issuer name"],
    "ticker": ["ticker", "holding ticker", "issuer ticker", "symbol", "local ticker"],
    "isin": ["isin"],
    "asset_class": ["asset class", "class of shares", "security type", "type", "asset type"],
    "weight_pct": ["weight (%)", "weight", "% of net assets", "portfolio weight",
                   "% weight", "weighting", "market value weight",
                   "percent of fund", "% of fund"],
}

# Normalized asset-class vocabulary. Order matters: first substring match wins, so
# more specific keys come first.
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

# Names that betray a cash / FX line even when the asset-class column is absent.
_CASH_NAME_RE = re.compile(
    r"\b(cash|usd|eur|gbp|collateral|margin|net other assets|fx|receivable|payable)\b",
    re.IGNORECASE,
)


def _find_column(cols: list[str], aliases: list[str]) -> str | None:
    """Return the first source column whose lowercased name matches an alias."""
    lower = {c.lower().strip(): c for c in cols}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def _normalize_asset_class(raw: object, name: str) -> str:
    """Map a raw asset-class string (or a telling name) to the small vocabulary."""
    text = str(raw).lower() if raw is not None and not pd.isna(raw) else ""
    for needle, label in _ASSET_CLASS_RULES:
        if needle in text:
            return label
    # No usable asset-class cell — fall back to the name (catch cash/FX lines).
    if _CASH_NAME_RE.search(str(name or "")):
        return "cash"
    return "equity"  # holdings rows are equities by default; flagged if it misreconciles


def _parse_weight(series: pd.Series) -> pd.Series:
    """Coerce a weight column to a 0..1 fraction (strips %, handles percent scale).

    Scale is detected by the column *sum*, not the max: a holdings file's weights sum
    to ~1 (fractions) or ~100 (percent). Using the max breaks for broad funds where no
    single holding exceeds 1.5% (e.g. Russell 2000) — the whole column would sum to ~100
    yet never get divided. The sum cleanly separates the two regimes.
    """
    s = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    vals = pd.to_numeric(s, errors="coerce")
    total = vals.abs().sum(skipna=True)
    if total is not None and total > 2.0:   # sums to ~100 -> percent scale
        vals = vals / 100.0
    return vals


def normalize_holdings(raw: pd.DataFrame, fund_id: str) -> pd.DataFrame:
    """Normalize one issuer's raw holdings frame to :data:`HOLDING_COLUMNS`.

    Missing optional columns (ticker/isin/asset_class) are tolerated — the resolver
    downstream degrades gracefully (ISIN -> ticker -> name). A missing *weight*
    column is fatal (raised): without weights there is no look-through.
    """
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

    # Drop lines with no usable weight (blank footer rows etc.), keep everything else.
    out = out[out["weight_in_fund"].notna()].reset_index(drop=True)
    missing = [f for f in ("ticker", "isin", "asset_class") if picked[f] is None]
    if missing:
        logger.info("%s: holdings CSV missing %s — resolver will lean on remaining ids",
                    fund_id, missing)
    return out[HOLDING_COLUMNS]
