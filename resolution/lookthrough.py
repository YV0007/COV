# Phase 3, step 3 — the look-through core: resolve, decompose, aggregate, reconcile

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("cov.resolution.lookthrough")

TREATMENTS = {"equity", "commodity", "crypto", "not_a_fund"}
_RESIDUAL_SUFFIX = "::cash_other"


def load_classification(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        logger.warning("No fund classification at %s — every fund left un-decomposed", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for treatment, ids in data.items():
        if treatment not in TREATMENTS:
            raise ValueError(f"Unknown treatment {treatment!r} in {path} (expected {TREATMENTS})")
        for cid in ids or []:
            out[cid] = treatment
    return out


def classify_funds(book: pd.DataFrame, classification: dict[str, str]) -> tuple[dict, list[str]]:
    treatments: dict[str, str] = {}
    unclassified: list[str] = []
    for r in book[book["is_fund"]].itertuples(index=False):
        t = classification.get(r.canonical_id)
        if t is None:
            unclassified.append(r.canonical_id)
        else:
            treatments[r.canonical_id] = t
    if unclassified:
        logger.warning("%d flagged fund(s) missing from classification (left un-decomposed): %s",
                       len(unclassified), unclassified)
    return treatments, unclassified


def _slug(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(text).upper())


def build_lookups(book: pd.DataFrame) -> dict:
    by_isin: dict[str, str] = {}
    by_ticker: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for r in book.itertuples(index=False):
        cid = r.canonical_id
        for isin in str(r.isins or "").split(";"):
            isin = isin.strip().upper()
            if isin:
                by_isin.setdefault(isin, cid)
        if r.ticker:
            by_ticker.setdefault(str(r.ticker).strip().upper(), cid)
        if r.name:
            by_name.setdefault(_slug(r.name), cid)
    return {"isin": by_isin, "ticker": by_ticker, "name": by_name}


def resolve_constituent(name, ticker, isin, lookups, overrides) -> tuple[str, str]:
    tkr = str(ticker).strip().upper() if ticker and not pd.isna(ticker) else ""
    isn = str(isin).strip().upper() if isin and not pd.isna(isin) else ""
    nm = _slug(name) if name and not pd.isna(name) else ""

    if tkr and tkr in overrides:
        return overrides[tkr], "override"
    if isn and isn in overrides:
        return overrides[isn], "override"
    if isn and isn in lookups["isin"]:
        return lookups["isin"][isn], "isin"
    if tkr and tkr in lookups["ticker"]:
        return lookups["ticker"][tkr], "ticker"
    if nm and nm in lookups["name"]:
        return lookups["name"][nm], "name"
    if isn:
        return f"x:{isn}", "synthetic_isin"
    if tkr:
        return f"x:{tkr}", "synthetic_ticker"
    return f"x:{nm}", "synthetic_name"


def build_effective_exposures(
    book: pd.DataFrame,
    holdings: pd.DataFrame,
    treatments: dict[str, str],
    overrides: dict[str, str] | None = None,
    recon_tol: float = 0.02,
    proxy_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overrides = {str(k).upper(): v for k, v in (overrides or {}).items()}
    proxy_map = proxy_map or {}
    lookups = build_lookups(book)
    book_weight = dict(zip(book["canonical_id"], book["weight"]))

    direct: dict[str, float] = defaultdict(float)
    indirect: dict[str, float] = defaultdict(float)
    via: dict[str, set] = defaultdict(set)
    names: dict[str, str] = {}
    residual_ids: set[str] = set()
    recon_rows: list[dict] = []

    for r in book.itertuples(index=False):
        cid = r.canonical_id
        if treatments.get(cid) == "equity":
            continue
        direct[cid] += float(r.weight)
        names.setdefault(cid, r.name)

    holdings_by_fund = dict(tuple(holdings.groupby("fund_id"))) if not holdings.empty else {}
    for fund_id, treatment in treatments.items():
        if treatment != "equity":
            continue
        w_e = float(book_weight.get(fund_id, 0.0))
        h = holdings_by_fund.get(fund_id)
        if h is None or h.empty:
            direct[fund_id] += w_e
            names.setdefault(fund_id, _book_name(book, fund_id))
            recon_rows.append({"fund_id": fund_id, "book_weight": w_e, "looked_through": False,
                               "constituent_weight_sum": None, "cash_residual": None,
                               "reconciled": False, "proxied_from": proxy_map.get(fund_id)})
            continue

        equity_h = h[h["asset_class"] == "equity"]
        eq_sum = float(equity_h["weight_in_fund"].sum())
        total_sum = float(h["weight_in_fund"].sum())
        for c in equity_h.itertuples(index=False):
            ccid, _method = resolve_constituent(c.constituent_name, c.ticker, c.isin,
                                                lookups, overrides)
            indirect[ccid] += w_e * float(c.weight_in_fund)
            via[ccid].add(fund_id)
            names.setdefault(ccid, c.constituent_name)

        residual = w_e * (1.0 - eq_sum)
        resid_cid = f"{fund_id}{_RESIDUAL_SUFFIX}"
        direct[resid_cid] += residual
        residual_ids.add(resid_cid)
        names[resid_cid] = f"{fund_id} cash & derivatives"
        recon_rows.append({
            "fund_id": fund_id, "book_weight": w_e, "looked_through": True,
            "constituent_weight_sum": round(total_sum, 6),
            "cash_residual": round(1.0 - eq_sum, 6),
            "reconciled": abs(total_sum - 1.0) <= recon_tol,
            "proxied_from": proxy_map.get(fund_id),
        })

    all_ids = set(direct) | set(indirect)
    rows = []
    for cid in all_ids:
        d, i = direct.get(cid, 0.0), indirect.get(cid, 0.0)
        rows.append({
            "canonical_id": cid,
            "name": names.get(cid),
            "direct_weight": d,
            "indirect_weight": i,
            "true_weight": d + i,
            "via_funds": ";".join(sorted(via.get(cid, set()))),
            "is_residual": cid in residual_ids,
        })
    exposures = (pd.DataFrame(rows)
                 .sort_values("true_weight", ascending=False)
                 .reset_index(drop=True))
    fund_recon = pd.DataFrame(recon_rows)
    return exposures, fund_recon


def _book_name(book: pd.DataFrame, cid: str) -> str | None:
    hit = book.loc[book["canonical_id"] == cid, "name"]
    return hit.iloc[0] if len(hit) else None


def reconcile_total(exposures: pd.DataFrame, book: pd.DataFrame, tol: float = 1e-6) -> dict:
    total_true = float(exposures["true_weight"].sum())
    total_book = float(book["weight"].sum())
    ok = abs(total_true - total_book) <= tol
    (logger.info if ok else logger.error)(
        "Look-through reconciliation: Σ true_weight %.6f vs Σ book_weight %.6f (%s)",
        total_true, total_book, "OK" if ok else "LEAK",
    )
    return {"total_true_weight": total_true, "total_book_weight": total_book, "reconciled": ok}


def concentration_report(exposures: pd.DataFrame, top_n: int = 15) -> dict:
    names_only = exposures[~exposures["is_residual"]].copy()

    def _hhi(w: pd.Series) -> float:
        s = w[w > 0]
        return float((s ** 2).sum())

    hhi_direct = _hhi(names_only["direct_weight"])
    hhi_true = _hhi(names_only["true_weight"])

    top_true = names_only.nlargest(top_n, "true_weight")[
        ["canonical_id", "name", "direct_weight", "indirect_weight", "true_weight"]
    ].copy()
    top_true["hidden_delta"] = top_true["true_weight"] - top_true["direct_weight"]

    hidden = names_only[(names_only["direct_weight"] <= 1e-9) & (names_only["indirect_weight"] > 0)]
    hidden = hidden.nlargest(top_n, "true_weight")[["canonical_id", "name", "true_weight"]]

    metrics = {
        "n_names_declared": int((names_only["direct_weight"] > 1e-9).sum()),
        "n_names_true": int((names_only["true_weight"] > 1e-9).sum()),
        "hhi_declared": round(hhi_direct, 6),
        "hhi_true": round(hhi_true, 6),
        "effective_names_declared": round(1.0 / hhi_direct, 1) if hhi_direct else None,
        "effective_names_true": round(1.0 / hhi_true, 1) if hhi_true else None,
        "top10_declared_weight": round(float(
            names_only.nlargest(10, "direct_weight")["direct_weight"].sum()), 6),
        "top10_true_weight": round(float(
            names_only.nlargest(10, "true_weight")["true_weight"].sum()), 6),
    }
    return {"top_true": top_true, "hidden_names": hidden, "metrics": metrics}
