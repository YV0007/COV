"""Phase 2, step 4 — per-instrument data-quality gates.

Nothing is dropped; everything is scored and *flagged* so later phases can decide.
Four gates (thresholds in config ``prices.quality_gates`` / ``min_history_days``):

  * **history** — fewer than ``min_history_days`` observations (short names: ARM, ETHA…).
  * **missing** — >max_missing_pct of business days in the covered span have no price.
  * **stale**   — a run of >= stale_price_days identical consecutive USD closes.
  * **outliers**— returns beyond ``outlier_sigma`` of a rolling 21-day mean/std.

Output matches the ``data_quality_flags`` DuckDB schema.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("cov.engine.quality")


def _max_identical_run(s: pd.Series) -> int:
    """Longest run of identical consecutive values (stale-price detector)."""
    if s.empty:
        return 0
    changed = s.ne(s.shift()).cumsum()
    return int(s.groupby(changed).transform("size").max())


def quality_report(
    prices_usd: pd.DataFrame,
    returns_long: pd.DataFrame,
    run_id: str,
    gates,
    min_history_days: int,
) -> pd.DataFrame:
    """One row per instrument matching ``data_quality_flags`` (+ a ``reasons`` note)."""
    max_missing = float(gates.max_missing_pct)
    stale_days = int(gates.stale_price_days)
    sigma = float(gates.outlier_sigma)
    win = int(gates.outlier_rolling_window)

    rows = []
    ret_by_id = dict(tuple(returns_long.groupby("canonical_id")))
    px_by_id = dict(tuple(prices_usd.groupby("canonical_id")))

    for cid, px in px_by_id.items():
        px = px.sort_values("date")
        r = ret_by_id.get(cid, pd.DataFrame(columns=["date", "return"]))
        history_days = len(r)

        # missing days vs a business-day calendar over the covered span
        if len(px) >= 2:
            span = pd.bdate_range(px["date"].min(), px["date"].max())
            # clamp at 0: crypto trades 7d/wk so obs can exceed business days
            missing_pct = round(max(0.0, 100.0 * (1 - len(px) / max(len(span), 1))), 2)
        else:
            missing_pct = 100.0

        stale_run = _max_identical_run(px["close_usd"].dropna())
        stale = stale_run >= stale_days

        if history_days >= win + 1:
            ret = r.sort_values("date")["return"]
            mu = ret.rolling(win).mean()
            sd = ret.rolling(win).std()
            z = (ret - mu) / sd
            outliers = int((z.abs() > sigma).sum())
        else:
            outliers = 0

        reasons = []
        if history_days < min_history_days:
            reasons.append(f"short_history({history_days}d)")
        if missing_pct > max_missing:
            reasons.append(f"missing({missing_pct}%)")
        if stale:
            reasons.append(f"stale_run({stale_run})")
        if px["close_usd"].isna().any():
            reasons.append("no_usd_price(fx_gap)")
        flagged = bool(reasons)

        rows.append({
            "run_id": run_id, "instrument_id": cid,
            "missing_pct": missing_pct, "stale": stale,
            "outlier_count": outliers, "history_days": history_days,
            "flagged": flagged, "reasons": ";".join(reasons),
        })

    report = pd.DataFrame(rows)
    n_flag = int(report["flagged"].sum()) if not report.empty else 0
    logger.info("Quality: %d/%d instruments flagged", n_flag, len(report))
    return report
