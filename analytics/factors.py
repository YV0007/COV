"""Phase 6 — factor overlay (the economic-naming layer).

Phase 5 found the *data-driven* factors and named them by eyeballing loadings ("looks
like AI-growth"). Phase 6 gives them **rigorous economic names** by regressing returns
against *named* factors — market, style, and macro — and reports the portfolio's net
exposure to each.

Factor sources:
  internal (from the Phase 2 returns, no network):
    MKT           = ACWI return                 (the market)
    GROWTH_VALUE  = VONG - QDIV return           (growth minus dividend/value tilt)
    CRYPTO        = BTC return
  external (FRED, behind a fetcher so tests stay offline):
    RATES         = daily change in the 10Y yield (DGS10)      — the 'duration' factor
    CREDIT        = daily change in HY OAS (BAMLH0A0HYM2)
    USD           = daily log-return of the broad USD index (DTWEXBGS)
    OIL           = daily log-return of WTI (DCOILWTICO)

Then: OLS each holding on the factors (betas + R²), weight-sum to the portfolio's net
exposure, and regress each Phase-5 PCA factor on the named set to label it economically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger("cov.analytics.factors")

# FRED codes for the macro block (name -> (code, transform)). 'diff' = level change,
# 'logret' = log return. Matches config.factors.macro.
_MACRO = {
    "RATES":  ("DGS10", "diff"),
    "CREDIT": ("BAMLH0A0HYM2", "diff"),
    "USD":    ("DTWEXBGS", "logret"),
    "OIL":    ("DCOILWTICO", "logret"),
}


# --- external factor fetcher (abstracted like PriceFetcher) -------------------
class FactorFetcher(ABC):
    """Fetch raw macro series keyed by name. Swap to change the source / go offline."""

    @abstractmethod
    def fetch(self, codes: dict, start: str, end: str) -> pd.DataFrame:
        """Return a DataFrame of raw level series (columns = the dict keys)."""


class FredFactorFetcher(FactorFetcher):
    def fetch(self, codes: dict, start: str, end: str) -> pd.DataFrame:
        try:
            import pandas_datareader.data as web
        except Exception:  # noqa: BLE001
            logger.warning("pandas_datareader unavailable — skipping macro factors")
            return pd.DataFrame()
        out = {}
        for name, code in codes.items():
            try:
                out[name] = web.DataReader(code, "fred", start, end)[code]
            except Exception as exc:  # noqa: BLE001 - one series failing must not kill it
                logger.warning("FRED fetch failed for %s (%s): %s", name, code, exc)
        return pd.DataFrame(out)


# --- factor panel -------------------------------------------------------------
def build_factor_panel(returns: pd.DataFrame, fetcher: FactorFetcher | None,
                       start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Assemble the (dates × factors) return panel; report each factor's source.

    Internal factors come free from the returns matrix; macro factors are fetched and
    transformed (diff / log-return). Any factor whose proxy is absent is skipped and
    reported — never silently zero-filled.
    """
    cols, sources = {}, {}
    # market proxy: a broad-market ETF if held, else the equal-weight mean of the book
    for proxy in ("ACWI", "SPY", "VOO", "VTI", "IWDA"):
        if proxy in returns.columns:
            cols["MKT"] = returns[proxy]; sources["MKT"] = proxy; break
    else:
        cols["MKT"] = returns.mean(axis=1); sources["MKT"] = "equal-weight mean"
    # growth-minus-value tilt, if both style proxies are held
    for g, v in (("VONG", "QDIV"), ("IWF", "IWD"), ("VUG", "VTV")):
        if g in returns.columns and v in returns.columns:
            cols["GROWTH_VALUE"] = returns[g] - returns[v]; sources["GROWTH_VALUE"] = f"{g}-{v}"; break
    # crypto proxy
    for c in ("BTC", "IBIT", "BITO"):
        if c in returns.columns:
            cols["CRYPTO"] = returns[c]; sources["CRYPTO"] = c; break

    if fetcher is not None:
        raw = fetcher.fetch({k: v[0] for k, v in _MACRO.items()}, start, end)
        for name, (_code, how) in _MACRO.items():
            if name in raw.columns:
                s = pd.to_numeric(raw[name], errors="coerce")
                cols[name] = s.diff() if how == "diff" else np.log(s).diff()
                sources[name] = f"FRED:{_MACRO[name][0]}"

    panel = pd.DataFrame(cols)
    panel.index = pd.to_datetime(panel.index)
    return panel, sources


# --- OLS: holdings on factors -------------------------------------------------
def regress_holdings(returns: pd.DataFrame, factors: pd.DataFrame,
                     min_obs: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """OLS each instrument's returns on the factor panel. Returns (betas, R²).

    betas: instrument × factor sensitivity; R²: factor-explained fraction of variance
    (1 − R² is idiosyncratic). Per-instrument date alignment handles differing history.
    """
    f = factors.copy()
    f.index = pd.to_datetime(f.index)
    betas, r2 = {}, {}
    for cid in returns.columns:
        y = returns[cid].copy()
        y.index = pd.to_datetime(y.index)
        df = pd.concat([y.rename("y"), f], axis=1, join="inner").dropna()
        if len(df) < min_obs:
            continue
        yv = df["y"].to_numpy()
        fx = df[f.columns].to_numpy()
        x = np.column_stack([np.ones(len(df)), fx])
        coef, *_ = np.linalg.lstsq(x, yv, rcond=None)
        resid = yv - x @ coef
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        r2[cid] = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
        betas[cid] = dict(zip(f.columns, coef[1:]))
    return pd.DataFrame(betas).T.reindex(columns=factors.columns), pd.Series(r2, name="r2")


def portfolio_exposure(betas: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Portfolio net factor exposure = book-weighted sum of holding betas.

    Weights are renormalized over the holdings that got a regression (coverage matters).
    Returns a per-factor table: net beta + the share of book weight behind it.
    """
    w = weights.reindex(betas.index).fillna(0.0)
    covered = float(w.sum())
    wn = w / covered if covered else w
    net = betas.mul(wn, axis=0).sum()
    return pd.DataFrame({"net_beta": net.round(4)}).assign(
        weight_covered=round(covered, 4)).sort_values("net_beta", key=lambda s: s.abs(),
                                                       ascending=False)


# --- name the Phase-5 factors -------------------------------------------------
def pca_factor_returns(detoned: pd.DataFrame, block: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Return series of the top-k PCA factors: f_j(t) = block · eigenvector_j."""
    from engine.covariance import eigidecompose
    common = [c for c in detoned.index if c in block.columns]
    d = detoned.loc[common, common]
    _vals, vecs = eigidecompose(d)
    b = block[common]
    out = {f"F{j+1}": b.to_numpy() @ vecs[:, j] for j in range(min(k, vecs.shape[1]))}
    return pd.DataFrame(out, index=block.index)


def name_factors(pca_returns: pd.DataFrame, factors: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Regress each PCA factor on the named factors; label it by its strongest loadings.

    Uses standardized factors so betas are comparable; the top-|beta| named factors (with
    the sign) become the economic name. Reports R² = how 'nameable' the factor is.
    """
    f = factors.copy(); f.index = pd.to_datetime(f.index)
    fz = (f - f.mean()) / f.std(ddof=0)
    rows = []
    for col in pca_returns.columns:
        y = pca_returns[col].copy(); y.index = pd.to_datetime(y.index)
        df = pd.concat([y.rename("y"), fz], axis=1, join="inner").dropna()
        if len(df) < 60:
            continue
        yv = (df["y"] - df["y"].mean()).to_numpy()
        x = df[fz.columns].to_numpy()
        coef, *_ = np.linalg.lstsq(x, yv, rcond=None)
        resid = yv - x @ coef
        r2 = 1.0 - float((resid ** 2).sum()) / float((yv ** 2).sum()) if (yv ** 2).sum() else np.nan
        s = pd.Series(coef, index=fz.columns)
        lead = s.reindex(s.abs().sort_values(ascending=False).index).head(top)
        label = ", ".join(f"{'+' if v > 0 else '-'}{n}" for n, v in lead.items())
        rows.append({"pca_factor": col, "economic_name": label, "r2": round(r2, 3)})
    return pd.DataFrame(rows)
