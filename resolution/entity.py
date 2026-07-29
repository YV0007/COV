"""Phase 1, step 2 — entity resolution: raw row -> canonical instrument.

The merge key is NOT raw ISIN or raw ticker alone; it is a hand-maintained table
(``config/entity_overrides.yaml``) layered on top of a default rule:

    default merge key = ISIN when present, else the normalized ticker.
    canonical_id      = a readable ticker slug per group.

Overrides handle the cases the default gets wrong:
  * **merges**   — several ISINs/tickers are one exposure (ASML dual line; IBIT+CBTC -> BTC).
  * **renames**  — relabel a slug to break a collision (BAE vs Boeing both ticker 'BA').
  * **keep_separate** — assert two ISINs must never merge (BMW ord vs BMW3 pref).

A **collision guard** fires if two genuinely different instruments (distinct ISINs)
land on the same canonical_id without a declared merge — the spec's "surface, don't
silently fix" rule applied to identity itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("cov.resolution.entity")


class ResolutionError(RuntimeError):
    """Raised when identity cannot be resolved safely (collision / bad merge)."""


def load_overrides(path: str | Path) -> dict:
    """Load the hand-maintained override table; tolerate a missing/empty file."""
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
    """Flatten overrides into fast per-row lookups.

    Returns (isin_merge, ticker_merge, isin_rename, merges) where the first three
    map a key -> the canonical spec dict, and ``merges`` is the raw merge mapping
    (used to know each merge's declared ISIN set for the collision guard).
    """
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
    """Assign ``canonical_id`` (+ canonical name/type/note) to every raw row.

    Adds columns: ``canonical_id``, ``canonical_name``, ``instrument_type``,
    ``shares_additive``, ``resolution_note``. Runs the keep-separate assertion and
    the collision guard before returning.
    """
    isin_merge, ticker_merge, isin_rename, merges = _build_lookups(overrides)
    df = df.copy()

    canonical_id, canonical_name, itype, shares_add, note = [], [], [], [], []
    for row in df.itertuples(index=False):
        isin = None if pd.isna(row.isin) else row.isin
        spec = None
        # 1. explicit merge (by ISIN, then by ticker for blank-ISIN wrappers)
        if isin and isin in isin_merge:
            spec = isin_merge[isin]
        elif row.norm_ticker in ticker_merge:
            spec = ticker_merge[row.norm_ticker]
        # 2. rename (disambiguate a slug collision)
        elif isin and isin in isin_rename:
            spec = isin_rename[isin]

        if spec is not None:
            canonical_id.append(spec["canonical_id"])
            canonical_name.append(spec.get("canonical_name", row.name))
            itype.append(spec.get("instrument_type", "equity"))
            shares_add.append(bool(spec.get("shares_additive", True)))
            note.append(spec.get("note", ""))
        else:
            # 3. default: readable ticker slug; group key handled via (id, isin) later
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
    """Fail loudly if two ISINs declared 'keep separate' share a canonical_id."""
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
    """Guard: one canonical_id must not span >1 distinct ISIN unless declared.

    For each canonical_id, collect the set of distinct non-null ISINs. Size <= 1 is
    always fine. Size > 1 is only allowed when it exactly matches (a subset of) a
    declared merge's ISIN set — otherwise it is an accidental ticker collision.
    """
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
