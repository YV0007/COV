# Phase 2, step 3 — convert prices to USD and compute log returns

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.fetch import PriceFetcher, YFinanceFetcher

logger = logging.getLogger("cov.engine.returns")

_PENCE = {"GBP": {"GBp", "GBX"}, "ZAR": {"ZAc"}, "ILS": {"ILA", "ILc"}}


def _major_and_divisor(currency: str) -> tuple[str, float]:
    for major, minors in _PENCE.items():
        if currency in minors:
            return major, 100.0
    return currency, 1.0


def _fx_series(currency: str, start: str, end: str, fetcher: PriceFetcher) -> pd.Series | None:
    for sym, invert in ((f"{currency}USD=X", False), (f"USD{currency}=X", True)):
        df, _ = fetcher.fetch_one(sym, start, end)
        if not df.empty:
            s = df.set_index("date")["close"].sort_index()
            s = 1.0 / s if invert else s
            return s
    return None


def to_usd(prices: pd.DataFrame, start: str, end: str,
           fetcher: PriceFetcher | None = None) -> tuple[pd.DataFrame, list[str]]:
    fetcher = fetcher or YFinanceFetcher()
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    majors = {c: _major_and_divisor(c)[0] for c in prices["currency"].unique()}
    divisors = {c: _major_and_divisor(c)[1] for c in prices["currency"].unique()}
    prices["major"] = prices["currency"].map(majors)
    prices["divisor"] = prices["currency"].map(divisors)

    all_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    prices["rate"] = 1.0
    missing: list[str] = []
    for major in sorted(m for m in majors.values() if m != "USD"):
        s = _fx_series(major, start, end, fetcher)
        mask = prices["major"] == major
        if s is None:
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
    p = prices.sort_values(["canonical_id", "date"]).copy()
    p["return"] = p.groupby("canonical_id")[price_col].transform(lambda s: np.log(s / s.shift(1)))
    return p.dropna(subset=["return"])[["date", "canonical_id", "return"]]


def returns_matrix(returns_long: pd.DataFrame) -> pd.DataFrame:
    if returns_long.empty:
        return pd.DataFrame()
    return returns_long.pivot_table(index="date", columns="canonical_id", values="return")
