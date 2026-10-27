# Phase 3, step 1 — fetch issuer ETF-holdings files (behind an interface)

from __future__ import annotations

import csv
import io
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("cov.resolution.holdings_sources")

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_NAME_TOKENS = ("name", "holding name", "security name", "description")
_WEIGHT_TOKENS = ("weight", "% of net assets", "% weight", "weighting",
                  "percent of fund", "% of fund")


def load_holdings_sources(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        logger.warning("No holdings-sources registry at %s — no funds can be fetched", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = data.get("sources", data) or {}
    return {k: v for k, v in sources.items()
            if isinstance(v, dict) and (v.get("url") or v.get("path") or v.get("proxy_of"))}


def _find_header_row(rows: list[list]) -> int:
    for i, row in enumerate(rows[:60]):
        cells = [str(c).strip().lower() for c in row]
        has_name = any(any(t in c for t in _NAME_TOKENS) for c in cells)
        has_weight = any(any(t in c for t in _WEIGHT_TOKENS) for c in cells)
        if has_name and has_weight:
            return i
    return 0


class HoldingsFetcher(ABC):
    @abstractmethod
    def fetch_one(self, fund_id: str, source: dict) -> pd.DataFrame:
        ...


class IssuerHoldingsFetcher(HoldingsFetcher):
    def __init__(self, timeout: int = 30, asof: str | None = None, project_root: Path | None = None) -> None:
        self.timeout = timeout
        self.asof = asof
        self.project_root = project_root

    def fetch_one(self, fund_id: str, source: dict) -> pd.DataFrame:
        filetype = (source.get("filetype") or "csv").lower()
        try:
            if source.get("path"):
                content = self._read_local(source["path"])
            else:
                content = self._download(fund_id, source)
            if content is None:
                return pd.DataFrame()
            if filetype == "json":
                return self._read_vanguard_json(content)
            if filetype in ("xlsx", "xls"):
                return self._read_excel(content, filetype)
            return self._read_csv(content)
        except Exception as exc:  # noqa: BLE001 - one fund failing must not kill the run
            logger.warning("Holdings fetch/parse failed for %s: %s", fund_id, exc)
            return pd.DataFrame()

    def _headers(self, url: str) -> dict:
        h = {"User-Agent": _UA, "Accept": "*/*"}
        if "ishares.com" in url:
            h["Referer"] = "https://www.ishares.com/"
        return h

    def _download(self, fund_id: str, source: dict) -> bytes | None:
        import requests
        url = source["url"]
        candidates = self._date_urls(url) if "{date}" in url else [url]
        for u in candidates:
            resp = requests.get(u, timeout=self.timeout, headers=self._headers(u))
            if resp.status_code == 200 and resp.content:
                return resp.content
            logger.debug("%s: %s -> HTTP %s", fund_id, u, resp.status_code)
        logger.warning("%s: no candidate URL returned data (last tried %s)", fund_id, candidates[-1])
        return None

    def _date_urls(self, url: str) -> list[str]:
        base = datetime.strptime(self.asof, "%Y-%m-%d") if self.asof else datetime(2000, 1, 1)
        return [url.replace("{date}", (base - timedelta(days=d)).strftime("%Y%m%d"))
                for d in range(0, 8)]

    def _read_local(self, path: str) -> bytes:
        p = Path(path)
        if not p.is_absolute() and self.project_root is not None:
            p = self.project_root / p
        return p.read_bytes()

    def _read_csv(self, content: bytes) -> pd.DataFrame:
        text = content.decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(text)))
        hdr = _find_header_row(rows)
        return pd.read_csv(io.StringIO(text), skiprows=hdr, thousands=",")

    def _read_excel(self, content: bytes, filetype: str) -> pd.DataFrame:
        engine = "xlrd" if filetype == "xls" else "openpyxl"
        head = pd.read_excel(io.BytesIO(content), header=None, engine=engine, nrows=60)
        hdr = _find_header_row(head.values.tolist())
        return pd.read_excel(io.BytesIO(content), skiprows=hdr, engine=engine)

    @staticmethod
    def _read_vanguard_json(content: bytes) -> pd.DataFrame:
        import json
        data = json.loads(content.decode("utf-8", errors="replace"))
        entities = (data.get("fund") or {}).get("entity") or []
        return pd.DataFrame([{
            "Name": e.get("longName") or e.get("shortName"),
            "Ticker": e.get("ticker"),
            "ISIN": e.get("isin"),
            "Asset Class": e.get("secMainType") or e.get("secSubType") or "Equity",
            "Weight (%)": e.get("percentWeight"),
        } for e in entities])


def fetch_holdings(
    fund_ids: list[str],
    sources: dict[str, dict],
    fetcher: HoldingsFetcher,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    raw_by_fund: dict[str, pd.DataFrame] = {}
    misses: list[dict] = []
    for fid in fund_ids:
        src = sources.get(fid)
        if not src:
            misses.append({"fund_id": fid, "reason": "no_source"})
            continue
        raw = fetcher.fetch_one(fid, src)
        if raw is None or raw.empty:
            misses.append({"fund_id": fid, "reason": "empty_or_failed"})
            continue
        raw_by_fund[fid] = raw
    logger.info("Fetched holdings for %d/%d equity funds (%d without data)",
                len(raw_by_fund), len(fund_ids), len(misses))
    return raw_by_fund, misses
