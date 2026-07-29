"""Reports layer — Jinja templates and report generation.

Home of Phase 8 (the scheduled client-facing report). Stub until built.
"""

from __future__ import annotations

import logging

from reports.report import generate_report

logger = logging.getLogger("cov.reports")

__all__ = ["generate_report"]
