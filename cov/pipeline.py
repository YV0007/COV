# End-to-end pipeline orchestrator

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from config.loader import Config, ensure_directories, load_config
from cov.logging_setup import setup_logging
from cov.storage import Storage

from analytics import detect_structure, factor_overlay, risk_attribution_and_stress
from engine import build_covariance, build_returns_matrix
from reports import generate_report
from resolution import etf_lookthrough, resolve_and_net_positions

PhaseFn = Callable[[Config, "RunContext"], str]
PHASES: list[tuple[int, str, PhaseFn]] = [
    (1, "Resolve & net positions", resolve_and_net_positions),
    (2, "Price history & returns", build_returns_matrix),
    (3, "ETF look-through", etf_lookthrough),
    (4, "Covariance engine", build_covariance),
    (5, "Structure detection", detect_structure),
    (6, "Factor overlay (naming)", factor_overlay),
    (7, "Risk attribution & stress", risk_attribution_and_stress),
    (8, "Reporting & delivery", generate_report),
]


@dataclass
class RunContext:
    run_id: str
    started_at: datetime
    storage: Storage
    artifacts: dict[str, Any] = field(default_factory=dict)


def run_pipeline(config: Config | None = None) -> RunContext:
    config = config or load_config()
    ensure_directories(config)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    logger = setup_logging(config, run_id=run_id)
    storage = Storage(config)
    ctx = RunContext(run_id=run_id, started_at=datetime.now(), storage=storage)

    logger.info("=" * 70)
    logger.info("COV pipeline run %s starting", run_id)
    logger.info("Base currency: %s", config.base_currency)
    logger.info("Storage: Parquet system-of-record + DuckDB at %s", storage.warehouse_path)
    logger.info("=" * 70)

    results: list[tuple[int, str, str]] = []
    for number, name, fn in PHASES:
        logger.info("--- Phase %d: %s ---", number, name)
        try:
            status = fn(config, ctx)
        except Exception:  # noqa: BLE001 - log full trace, keep the run auditable
            logger.exception("Phase %d (%s) failed", number, name)
            status = "error"
        results.append((number, name, status))

    logger.info("=" * 70)
    logger.info("Run %s summary:", run_id)
    for number, name, status in results:
        logger.info("  Phase %d — %-28s %s", number, name, status)
    logger.info("Run %s complete.", run_id)
    logger.info("=" * 70)

    ctx.artifacts["results"] = results
    storage.close()
    return ctx


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
