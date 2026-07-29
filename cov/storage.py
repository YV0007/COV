"""The single storage layer — the ONLY module in the project that touches disk.

Design (see README §Storage):
  * **Parquet files** on disk are the append-only *system of record* — durable,
    portable, auditable, reproducible. Nothing important lives only in the database.
  * **DuckDB** (one ``warehouse.duckdb`` file) is the *query engine* over those Parquet
    files, and the home for small hand-edited relational reference tables (entity map,
    instrument metadata, run metadata, data-quality flags).
  * Reads go *through DuckDB* even for Parquet, so a returns panel can be joined to the
    entity map / sector labels in one SQL statement.
  * Writes **append** a new Parquet file/partition — never overwrite. Reference-table
    edits happen in DuckDB.

Everything else in the codebase calls this module (``get_returns``, ``write_snapshot``,
``get_entity_map`` …). That indirection is what lets the storage engine or data source
be swapped later without touching the engine/analytics code.

At Phase 0 the plumbing is real and tested (connection, schema, append-only Parquet
writers, DuckDB-backed readers); the query *shapes* will grow as later phases need them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pandas as pd

from config.loader import Config, ensure_directories, load_config

logger = logging.getLogger("cov.storage")


# --- Reference-table schema (small, relational, lives inside DuckDB) -----------
# These are the ONLY tables that hold authoritative data inside the database; they are
# hand-edited or written each run and constantly joined against the Parquet panels.
_REFERENCE_SCHEMA: dict[str, str] = {
    # ISIN/ticker -> canonical instrument. Hand-built, grows with the book.
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
    # Sector labels (the 20-bucket taxonomy) + GICS-like reference per instrument.
    "instrument_metadata": """
        CREATE TABLE IF NOT EXISTS instrument_metadata (
            instrument_id   VARCHAR PRIMARY KEY,
            sector_label    VARCHAR,               -- client's self-declared bucket
            gics_sector     VARCHAR,               -- official-ish, for comparison
            listing_country VARCHAR,
            currency        VARCHAR
        )
    """,
    # One row per run: the provenance every computed matrix must carry.
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
    # Per-run, per-instrument data-quality flags (Phase 2 writes these).
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
    """Thin facade over Parquet (system of record) + DuckDB (query layer).

    Construct once per process and pass around, or use :func:`get_storage` for a
    config-defaulted instance. The DuckDB connection is opened lazily on first use and
    the reference-table schema is created if absent.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        ensure_directories(self.config)
        self._con: duckdb.DuckDBPyConnection | None = None

    # -- paths ---------------------------------------------------------------
    def _path(self, key: str) -> Path:
        return Path(self.config.paths[key])

    @property
    def warehouse_path(self) -> Path:
        return self._path("warehouse_duckdb")

    # -- connection ----------------------------------------------------------
    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        """Lazily open the DuckDB connection and ensure reference tables exist."""
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

    # ======================================================================
    # Reference tables (DuckDB) — hand-edited / joined constantly
    # ======================================================================
    def get_entity_map(self) -> pd.DataFrame:
        """Return the full ISIN/ticker -> canonical instrument map."""
        return self.con.execute("SELECT * FROM entity_map").fetchdf()

    def get_instrument_metadata(self) -> pd.DataFrame:
        """Return per-instrument sector labels and reference metadata."""
        return self.con.execute("SELECT * FROM instrument_metadata").fetchdf()

    def upsert_reference(self, table: str, df: pd.DataFrame) -> int:
        """Insert/replace rows in a reference table from a DataFrame.

        Reference-table edits are the one kind of write that mutates the DB in place
        (they are small and hand-curated). Parquet data is never mutated this way.
        """
        if table not in _REFERENCE_SCHEMA:
            raise ValueError(f"Unknown reference table: {table!r}")
        con = self.con
        con.register("_incoming", df)
        con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _incoming")
        con.unregister("_incoming")
        return len(df)

    def record_run_metadata(self, df: pd.DataFrame) -> int:
        """Append run-provenance rows (window, estimator, condition #, coverage …)."""
        con = self.con
        con.register("_incoming", df)
        con.execute("INSERT INTO run_metadata BY NAME SELECT * FROM _incoming")
        con.unregister("_incoming")
        return len(df)

    def record_quality_flags(self, df: pd.DataFrame) -> int:
        """Append per-run per-instrument data-quality flags (Phase 2)."""
        con = self.con
        con.register("_incoming", df)
        con.execute("INSERT INTO data_quality_flags BY NAME SELECT * FROM _incoming")
        con.unregister("_incoming")
        return len(df)

    # ======================================================================
    # Parquet writers (append-only system of record) — never overwrite
    # ======================================================================
    @staticmethod
    def _write_parquet(df: pd.DataFrame, target_dir: Path, filename: str) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / filename
        df.to_parquet(out, index=False)
        logger.info("Wrote %d rows -> %s", len(df), out)
        return out

    def write_prices(self, df: pd.DataFrame, year: int, name: str = "prices") -> Path:
        """Append a daily-price/returns Parquet file into ``prices/year=YYYY/``."""
        target = self._path("parquet_prices") / f"year={year}"
        return self._write_parquet(df, target, f"{name}.parquet")

    def write_raw_prices(self, df: pd.DataFrame, asof: str, name: str = "prices_raw") -> Path:
        """Cache an untouched price pull into ``raw/prices/asof=YYYY-MM-DD/`` (audit)."""
        target = self._path("raw_prices") / f"asof={asof}"
        return self._write_parquet(df, target, f"{name}.parquet")

    def write_holdings_snapshot(self, df: pd.DataFrame, asof: str, name: str = "holdings") -> Path:
        """Append an ETF-constituents snapshot into ``holdings/asof=YYYY-MM/``."""
        target = self._path("parquet_holdings") / f"asof={asof}"
        return self._write_parquet(df, target, f"{name}.parquet")

    def write_factors(self, df: pd.DataFrame, name: str) -> Path:
        """Write/append a factor or FX series into ``factors/`` (one file per series)."""
        return self._write_parquet(df, self._path("parquet_factors"), f"{name}.parquet")

    def write_snapshot(self, kind: str, run_date: str, df: pd.DataFrame, name: str) -> Path:
        """Write an immutable per-run analytical output.

        ``kind`` is ``"book"``, ``"covariance"`` or ``"exposure"`` ->
        ``snapshots/<kind>/run=YYYY-MM-DD/``. Given a ``run=`` folder, every downstream
        number is reproducible.
        """
        key = {"book": "snapshots_book", "covariance": "snapshots_covariance",
               "exposure": "snapshots_exposure"}.get(kind)
        if key is None:
            raise ValueError(
                f"Unknown snapshot kind: {kind!r} (expected 'book', 'covariance' or 'exposure')")
        target = self._path(key) / f"run={run_date}"
        return self._write_parquet(df, target, f"{name}.parquet")

    # ======================================================================
    # Parquet readers (through DuckDB) — bulk time-series
    # ======================================================================
    @staticmethod
    def _glob_has_files(root: Path) -> bool:
        return root.exists() and any(root.rglob("*.parquet"))

    def _read_parquet_tree(self, root: Path, where: str = "", params: Sequence | None = None) -> pd.DataFrame:
        """Read every Parquet file under ``root`` via DuckDB, empty frame if none."""
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
        """Load daily prices/returns from the Parquet system of record.

        Filters by ticker/date when given. Returns an empty frame when no price data
        has been written yet (Phase 2 populates it).
        """
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
        """Return a returns panel (dates x instruments).

        Thin wrapper over :meth:`get_prices` that pivots the ``return`` column into a
        wide matrix. Empty in, empty out. As Phase 4+ needs it, this is where the join
        to the entity map / sector labels will happen in one DuckDB statement.
        """
        df = self.get_prices(tickers, start, end)
        # Key the matrix on canonical_id when present (bare tickers collide, e.g.
        # Boeing vs BAE both 'BA'); fall back to ticker for legacy price frames.
        key = "canonical_id" if "canonical_id" in df.columns else "ticker"
        if df.empty or not {"date", key, "return"}.issubset(df.columns):
            return df
        return df.pivot_table(index="date", columns=key, values="return")

    def get_factors(self, names: Iterable[str] | None = None) -> pd.DataFrame:
        """Load factor / FX series from ``factors/`` (all, or a named subset)."""
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
        """Load a netted-book snapshot; latest ``run=`` folder if none specified."""
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
        """Load ETF-constituent holdings, optionally for a single ``asof=YYYY-MM``."""
        root = self._path("parquet_holdings")
        if asof is not None:
            root = root / f"asof={asof}"
        return self._read_parquet_tree(root)

    def get_covariance(self, name: str, run_date: str | None = None) -> pd.DataFrame:
        """Load a named covariance-snapshot file (latest ``run=`` folder if unspecified).

        ``name`` is the snapshot's base name, e.g. ``corr_full_detoned`` or
        ``diagnostics``. Used by Phase 5 to consume Phase 4's cleaned matrices.
        """
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
        # hive_partitioning=false: reading a single file inside run=.../ must NOT inject a
        # 'run' partition column into the matrix (it would break the eigen-decomposition).
        return self.con.execute(
            "SELECT * FROM read_parquet(?, hive_partitioning=false)", [str(f)]).fetchdf()


# --- module-level convenience -------------------------------------------------
_default: Storage | None = None


def get_storage(config: Config | None = None) -> Storage:
    """Return a process-wide default Storage (or a fresh one bound to ``config``)."""
    global _default
    if config is not None:
        return Storage(config)
    if _default is None:
        _default = Storage()
    return _default
