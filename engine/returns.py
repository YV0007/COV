"""Phase 2, step 3 — convert prices to USD and compute log returns.

Base currency is USD, so a USD investor's return in a foreign name includes the FX
move. We therefore convert every price to USD *before* differencing: ``close_usd =
close_local / divisor * fx_rate``. Two subtleties:

  * **GBp / GBX** (London pence): Yahoo quotes these in pence, so divisor = 100 to get
    GBP before applying GBP/USD. Missing this silently inflates UK names 100x.
  * FX pair convention: ``<CCY>USD=X`` is USD per 1 unit of CCY (multiply). If only
    ``USD<CCY>=X`` exists we invert it.

Returns are natural-log (additive over time); the wide matrix is dates × instruments.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.fetch import PriceFetcher, YFinanceFetcher

logger = logging.getLogger("cov.engine.returns")

# Minor-unit quotes -> (major currency, divisor 100). London pence (GBp/GBX),
# South-African cents (ZAc), Israeli agorot (ILA). Missing this inflates prices 100x.
_PENCE = {"GBP": {"GBp", "GBX"}, "ZAR": {"ZAc"}, "ILS": {"ILA", "ILc"}}


def _major_and_divisor(currency: str) -> tuple[str, float]:
    for major, minors in _PENCE.items():
        if currency in minors:
            return major, 100.0
    return currency, 1.0


def _fx_series(currency: str, start: str, end: str, fetcher: PriceFetcher) -> pd.Series | None:
    """USD-per-1-unit-of-``currency`` daily series (forward-filled). None on failure."""
    for sym, invert in ((f"{currency}USD=X", False), (f"USD{currency}=X", True)):
        df, _ = fetcher.fetch_one(sym, start, end)
        if not df.empty:
            s = df.set_index("date")["close"].sort_index()
            s = 1.0 / s if invert else s
            return s
    return None


def to_usd(prices: pd.DataFrame, start: str, end: str,
           fetcher: PriceFetcher | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Add a ``close_usd`` column. Returns ``(prices_with_usd, currencies_missing_fx)``."""
    fetcher = fetcher or YFinanceFetcher()
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    majors = {c: _major_and_divisor(c)[0] for c in prices["currency"].unique()}
    divisors = {c: _major_and_divisor(c)[1] for c in prices["currency"].unique()}
    prices["major"] = prices["currency"].map(majors)
    prices["divisor"] = prices["currency"].map(divisors)

    # One FX series per distinct non-USD major, aligned (ffilled) to all price dates.
    all_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    prices["rate"] = 1.0
    missing: list[str] = []
    for major in sorted(m for m in majors.values() if m != "USD"):
        s = _fx_series(major, start, end, fetcher)
        mask = prices["major"] == major
        if s is None:  # FX missing -> NaN (flagged upstream), never fabricated
            missing.append(major)
            prices.loc[mask, "rate"] = np.nan
            continue
        aligned = s.reindex(s.index.union(all_dates)).sort_index().ffill().reindex(all_dates)
        prices.loc[mask, "rate"] = prices.loc[mask, "date"].map(aligned)

    prices["close_usd"] = prices["close"] / prices["divisor"] * prices["rate"]
    prices = prices.drop(columns=["major", "divisor", "rate"])
    if missing:
        logger.warning("No FX series for %s — those names left un-converted (flagged)", missing)
    logger.info("Converted %d price rows to USD across %d currencies",
                len(prices), prices["currency"].nunique())
    return prices, missing


def log_returns(prices: pd.DataFrame, price_col: str = "close_usd") -> pd.DataFrame:
    """Per-instrument natural-log returns -> long ``[date, canonical_id, return]``."""
    p = prices.sort_values(["canonical_id", "date"]).copy()
    p["return"] = p.groupby("canonical_id")[price_col].transform(lambda s: np.log(s / s.shift(1)))
    return p.dropna(subset=["return"])[["date", "canonical_id", "return"]]


def returns_matrix(returns_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to the Phase 2 deliverable: dates × instruments."""
    if returns_long.empty:
        return pd.DataFrame()
    return returns_long.pivot_table(index="date", columns="canonical_id", values="return")
