"""Phase 2, step 2 — the price fetcher (behind an interface so sources can swap).

``PriceFetcher`` is the contract; ``YFinanceFetcher`` is the free default. yfinance is
a *library* over Yahoo's data API — not a scraper and nothing is hand-ingested. We pull
daily **auto-adjusted** close (splits + dividends folded in) per symbol, keeping the
listing currency (taken from Yahoo, never inferred) so FX conversion is correct later.

Fetches run in a small thread pool; a symbol that returns nothing is recorded as a
failure and surfaced — never silently dropped.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

logger = logging.getLogger("cov.engine.fetch")


class PriceFetcher(ABC):
    """Fetch daily adjusted close for one symbol. Swap this to change data source."""

    @abstractmethod
    def fetch_one(self, symbol: str, start: str, end: str) -> tuple[pd.DataFrame, str | None]:
        """Return ``(DataFrame[date, close], currency)``; empty frame if not found."""


class YFinanceFetcher(PriceFetcher):
    def fetch_one(self, symbol: str, start: str, end: str) -> tuple[pd.DataFrame, str | None]:
        import yfinance as yf

        t = yf.Ticker(symbol)
        h = t.history(start=start, end=end, auto_adjust=True)
        if h is None or h.empty:
            return pd.DataFrame(columns=["date", "close"]), None
        currency = None
        try:
            currency = t.fast_info.get("currency")
        except Exception:  # noqa: BLE001 - currency is best-effort
            pass
        df = h[["Close"]].rename(columns={"Close": "close"})
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df.reset_index(names="date")[["date", "close"]]
        return df, currency


def fetch_prices(
    symbols: pd.DataFrame,
    start: str,
    end: str,
    fetcher: PriceFetcher | None = None,
    max_workers: int = 6,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch every mapped symbol in ``symbols`` (skip ``kind == 'unmapped'``).

    ``symbols`` has columns ``canonical_id, ticker, yahoo_symbol, kind``. Returns
    ``(long_prices, failures)`` where ``long_prices`` has ``canonical_id, ticker,
    yahoo_symbol, currency, date, close`` and ``failures`` lists what came back empty.
    """
    fetcher = fetcher or YFinanceFetcher()
    todo = symbols[symbols["kind"] != "unmapped"].dropna(subset=["yahoo_symbol"])
    unmapped = symbols[symbols["kind"] == "unmapped"]

    frames: list[pd.DataFrame] = []
    failures: list[dict] = [
        {"canonical_id": r.canonical_id, "yahoo_symbol": None, "reason": "unmapped"}
        for r in unmapped.itertuples(index=False)
    ]

    def _job(row):
        df, ccy = fetcher.fetch_one(row.yahoo_symbol, start, end)
        return row, df, ccy

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_job, r) for r in todo.itertuples(index=False)]
        for fut in as_completed(futures):
            row, df, ccy = fut.result()
            if df.empty:
                failures.append({"canonical_id": row.canonical_id,
                                 "yahoo_symbol": row.yahoo_symbol, "reason": "no_data"})
                continue
            df = df.assign(canonical_id=row.canonical_id, ticker=row.ticker,
                           yahoo_symbol=row.yahoo_symbol, currency=ccy or "USD")
            frames.append(df)

    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["canonical_id", "ticker", "yahoo_symbol", "currency", "date", "close"])
    logger.info("Fetched %d/%d symbols (%d failed/unmapped)",
                prices["canonical_id"].nunique() if not prices.empty else 0,
                len(symbols), len(failures))
    return prices, failures
