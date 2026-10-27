# Reports layer — Jinja templates and report generation

from __future__ import annotations

import logging

from reports.report import generate_report

logger = logging.getLogger("cov.reports")

__all__ = ["generate_report"]
