"""COV — Portfolio Correlation & Exposure Model.

Top-level orchestration package. The phase logic itself lives in the domain packages
(``resolution``, ``engine``, ``analytics``, ``reports``); this package wires them into
one end-to-end batch run and owns cross-cutting concerns (logging, config plumbing).
"""

__version__ = "0.1.0"
