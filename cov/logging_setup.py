"""Logging configuration for pipeline runs.

Produces a run logger that writes both to the console and to a timestamped file under
the configured ``logs/`` directory, so every batch run leaves an auditable trail
(point-in-time discipline applies to logs too).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from config.loader import Config

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(config: Config, run_id: str | None = None) -> logging.Logger:
    """Configure and return the root ``cov`` logger for a run.

    A new timestamped log file is created per run (never overwritten). ``run_id`` is
    used in the filename when provided, otherwise a UTC timestamp is generated.
    """
    log_cfg = config.get("logging", None)
    level_name = getattr(log_cfg, "level", "INFO") if log_cfg else "INFO"
    to_file = getattr(log_cfg, "to_file", True) if log_cfg else True
    to_console = getattr(log_cfg, "to_console", True) if log_cfg else True
    level = getattr(logging, str(level_name).upper(), logging.INFO)

    logger = logging.getLogger("cov")
    logger.setLevel(level)
    logger.handlers.clear()  # idempotent: safe to call more than once per process
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if to_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)

    if to_file:
        logs_dir = Path(config.paths.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
        log_path = logs_dir / f"run_{stamp}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging to %s", log_path)

    return logger
