# Phase 1, step 2 — entity resolution: raw row -> canonical instrument

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("cov.resolution.entity")


class ResolutionError(RuntimeError):
    ...


def load_overrides(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        logger.warning("No entity_overrides file at %s — using default resolution only", path)
        return {"merges": {}, "renames": [], "keep_separate": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("merges", {})
    data.setdefault("renames", [])
    data.setdefault("keep_separate", [])
    return data


def _build_lookups(overrides: dict) -> tuple[dict, dict, dict, list]:
    merges = overrides.get("merges", {})
    isin_merge: dict[str, dict] = {}
    ticker_merge: dict[str, dict] = {}
    for spec in merges.values():
        for isin in spec.get("match_isin", []) or []:
            isin_merge[isin] = spec
        for tkr in spec.get("match_ticker", []) or []:
            ticker_merge[str(tkr).upper()] = spec

    isin_rename: dict[str, dict] = {}
    for spec in overrides.get("renames", []) or []:
        isin_rename[spec["match_isin"]] = spec
    return isin_merge, ticker_merge, isin_rename, merges


def resolve(df: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    isin_merge, ticker_merge, isin_rename, merges = _build_lookups(overrides)
    df = df.copy()

    canonical_id, canonical_name, itype, shares_add, note = [], [], [], [], []
    for row in df.itertuples(index=False):
        isin = None if pd.isna(row.isin) else row.isin
        spec = None
        if isin and isin in isin_merge:
            spec = isin_merge[isin]
        elif row.norm_ticker in ticker_merge:
            spec = ticker_merge[row.norm_ticker]
        elif isin and isin in isin_rename:
            spec = isin_rename[isin]

        if spec is not None:
            canonical_id.append(spec["canonical_id"])
            canonical_name.append(spec.get("canonical_name", row.name))
            itype.append(spec.get("instrument_type", "equity"))
            shares_add.append(bool(spec.get("shares_additive", True)))
            note.append(spec.get("note", ""))
        else:
            canonical_id.append(row.norm_ticker)
            canonical_name.append(row.name)
            itype.append("equity")
            shares_add.append(True)
            note.append("")

    df["canonical_id"] = canonical_id
    df["canonical_name"] = canonical_name
    df["instrument_type"] = itype
    df["shares_additive"] = shares_add
    df["resolution_note"] = note

    _assert_keep_separate(df, overrides)
    _guard_collisions(df, merges)
    logger.info("Resolved %d rows -> %d canonical instruments",
                len(df), df["canonical_id"].nunique())
    return df


def _assert_keep_separate(df: pd.DataFrame, overrides: dict) -> None:
    id_by_isin = (
        df.dropna(subset=["isin"]).groupby("isin")["canonical_id"].agg(lambda s: set(s))
    )
    for pair in overrides.get("keep_separate", []) or []:
        isins = pair.get("isins", [])
        ids: set[str] = set()
        for isin in isins:
            ids |= id_by_isin.get(isin, set())
        if len(ids) < 2 and len(isins) >= 2 and all(i in id_by_isin.index for i in isins):
            raise ResolutionError(
                f"keep_separate violated: ISINs {isins} collapsed to {ids} "
                f"({pair.get('reason', '')})"
            )


def _guard_collisions(df: pd.DataFrame, merges: dict) -> None:
    declared: dict[str, set[str]] = {
        spec["canonical_id"]: set(spec.get("match_isin", []) or [])
        for spec in merges.values()
    }
    isins_by_id = df.dropna(subset=["isin"]).groupby("canonical_id")["isin"].agg(
        lambda s: set(s)
    )
    for cid, isins in isins_by_id.items():
        if len(isins) <= 1:
            continue
        if isins <= declared.get(cid, set()):
            continue
        raise ResolutionError(
            f"canonical_id {cid!r} spans multiple ISINs {sorted(isins)} with no "
            f"declared merge — likely a ticker collision. Add a rename/merge to "
            f"config/entity_overrides.yaml."
        )
