# The single storage layer — the ONLY module in the project that touches disk

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pandas as pd

from config.loader import Config, ensure_directories, load_config

logger = logging.getLogger("cov.storage")


_REFERENCE_SCHEMA: dict[str, str] = {
    "entity_map": """
        CREATE TABLE IF NOT EXISTS entity_map (
            instrument_id   VARCHAR PRIMARY KEY,   -- canonical id you assign
            canonical_name  VARCHAR,
            isin            VARCHAR,
            ticker          VARCHAR,
            instrument_type VARCHAR,               -- equity | etf | etp | adr
            note            VARCHAR
        )
    """,
    "instrument_metadata": """
        CREATE TABLE IF NOT EXISTS instrument_metadata (
            instrument_id   VARCHAR PRIMARY KEY,
            sector_label    VARCHAR,               -- client's self-declared bucket
            gics_sector     VARCHAR,               -- official-ish, for comparison
            listing_country VARCHAR,
            currency        VARCHAR
        )
    """,
    "run_metadata": """
        CREATE TABLE IF NOT EXISTS run_metadata (
            run_id           VARCHAR,
            run_date         DATE,
            phase            VARCHAR,
            est_window       VARCHAR,               -- 'window' is a DuckDB reserved word
            estimator        VARCHAR,
            condition_number DOUBLE,
            effective_rank   DOUBLE,
            coverage_pct     DOUBLE,
            created_at       TIMESTAMP DEFAULT current_timestamp
        )
    """,
    "data_quality_flags": """
        CREATE TABLE IF NOT EXISTS data_quality_flags (
            run_id        VARCHAR,
            instrument_id VARCHAR,
            missing_pct   DOUBLE,
            stale         BOOLEAN,
            outlier_count INTEGER,
            history_days  INTEGER,
            flagged       BOOLEAN
        )
    """,
}


class Storage:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        ensure_directories(self.config)
        self._con: duckdb.DuckDBPyConnection | None = None

    def _path(self, key: str) -> Path:
        return Path(self.config.paths[key])

    @property
    def warehouse_path(self) -> Path:
        return self._path("warehouse_duckdb")

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self.warehouse_path.parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(str(self.warehouse_path))
            self._init_reference_tables(self._con)
            logger.info("Opened DuckDB warehouse at %s", self.warehouse_path)
        return self._con

    @staticmethod
    def _init_reference_tables(con: duckdb.DuckDBPyConnection) -> None:
        for ddl in _REFERENCE_SCHEMA.values():
            con.execute(ddl)

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_entity_map(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM entity_map").fetchdf()

    def get_instrument_metadata(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM instrument_metadata").fetchdf()

    def upsert_reference(self, table: str, df: pd.DataFrame) -> int:
        if table not in _REFERENCE_SCHEMA:
            raise ValueError(f"Unknown reference table: {table!r}")
        con = self.con
        con.register("_incoming", df)
        con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _incoming")
        con.unregister("_incoming")
        return len(df)

    def record_run_metadata(self, df: pd.DataFrame) -> int:
        con = self.con
        con.register("_incoming", df)
        con.execute("INSERT INTO run_metadata BY NAME SELECT * FROM _incoming")
        con.unregister("_incoming")
        return len(df)

    def record_quality_flags(self, df: pd.DataFrame) -> int:
        con = self.con
        con.register("_incoming", df)
        con.execute("INSERT INTO data_quality_flags BY NAME SELECT * FROM _incoming")
        con.unregister("_incoming")
        return len(df)

    @staticmethod
    def _write_parquet(df: pd.DataFrame, target_dir: Path, filename: str) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / filename
        df.to_parquet(out, index=False)
        logger.info("Wrote %d rows -> %s", len(df), out)
        return out

    def write_prices(self, df: pd.DataFrame, year: int, name: str = "prices") -> Path:
        target = self._path("parquet_prices") / f"year={year}"
        return self._write_parquet(df, target, f"{name}.parquet")

    def write_raw_prices(self, df: pd.DataFrame, asof: str, name: str = "prices_raw") -> Path:
        target = self._path("raw_prices") / f"asof={asof}"
        return self._write_parquet(df, target, f"{name}.parquet")

    def write_holdings_snapshot(self, df: pd.DataFrame, asof: str, name: str = "holdings") -> Path:
        target = self._path("parquet_holdings") / f"asof={asof}"
        return self._write_parquet(df, target, f"{name}.parquet")

    def write_factors(self, df: pd.DataFrame, name: str) -> Path:
        return self._write_parquet(df, self._path("parquet_factors"), f"{name}.parquet")

    def write_snapshot(self, kind: str, run_date: str, df: pd.DataFrame, name: str) -> Path:
        key = {"book": "snapshots_book", "covariance": "snapshots_covariance",
               "exposure": "snapshots_exposure"}.get(kind)
        if key is None:
            raise ValueError(
                f"Unknown snapshot kind: {kind!r} (expected 'book', 'covariance' or 'exposure')")
        target = self._path(key) / f"run={run_date}"
        return self._write_parquet(df, target, f"{name}.parquet")

    @staticmethod
    def _glob_has_files(root: Path) -> bool:
        return root.exists() and any(root.rglob("*.parquet"))

    def _read_parquet_tree(self, root: Path, where: str = "", params: Sequence | None = None) -> pd.DataFrame:
        if not self._glob_has_files(root):
            return pd.DataFrame()
        glob = str(root / "**" / "*.parquet")
        sql = f"SELECT * FROM read_parquet(?, hive_partitioning=true, union_by_name=true)"
        if where:
            sql += f" WHERE {where}"
        return self.con.execute(sql, [glob, *(params or [])]).fetchdf()

    def get_prices(
        self,
        tickers: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list = []
        if tickers is not None:
            tickers = list(tickers)
            placeholders = ", ".join(["?"] * len(tickers))
            clauses.append(f"ticker IN ({placeholders})")
            params.extend(tickers)
        if start is not None:
            clauses.append("date >= ?")
            params.append(start)
        if end is not None:
            clauses.append("date <= ?")
            params.append(end)
        where = " AND ".join(clauses)
        return self._read_parquet_tree(self._path("parquet_prices"), where, params)

    def get_returns(
        self,
        tickers: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        df = self.get_prices(tickers, start, end)
        key = "canonical_id" if "canonical_id" in df.columns else "ticker"
        if df.empty or not {"date", key, "return"}.issubset(df.columns):
            return df
        return df.pivot_table(index="date", columns=key, values="return")

    def get_factors(self, names: Iterable[str] | None = None) -> pd.DataFrame:
        root = self._path("parquet_factors")
        if names is None:
            return self._read_parquet_tree(root)
        frames = []
        for n in names:
            f = root / f"{n}.parquet"
            if f.exists():
                frames.append(self.con.execute("SELECT * FROM read_parquet(?)", [str(f)]).fetchdf())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_book(self, run_date: str | None = None) -> pd.DataFrame:
        root = self._path("snapshots_book")
        if run_date is None:
            runs = sorted(root.glob("run=*")) if root.exists() else []
            if not runs:
                return pd.DataFrame()
            root = runs[-1]
        else:
            root = root / f"run={run_date}"
        return self._read_parquet_tree(root)

    def get_holdings(self, asof: str | None = None) -> pd.DataFrame:
        root = self._path("parquet_holdings")
        if asof is not None:
            root = root / f"asof={asof}"
        return self._read_parquet_tree(root)

    def get_covariance(self, name: str, run_date: str | None = None) -> pd.DataFrame:
        root = self._path("snapshots_covariance")
        if run_date is None:
            runs = sorted(root.glob("run=*")) if root.exists() else []
            if not runs:
                return pd.DataFrame()
            root = runs[-1]
        else:
            root = root / f"run={run_date}"
        f = root / f"{name}.parquet"
        if not f.exists():
            return pd.DataFrame()
        return self.con.execute(
            "SELECT * FROM read_parquet(?, hive_partitioning=false)", [str(f)]).fetchdf()


_default: Storage | None = None


def get_storage(config: Config | None = None) -> Storage:
    global _default
    if config is not None:
        return Storage(config)
    if _default is None:
        _default = Storage()
    return _default
