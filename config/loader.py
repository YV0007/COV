"""Configuration loader.

Loads ``config/config.yaml`` into a lightweight, attribute-accessible object and
resolves all declared paths relative to the project root. Keeping this in one place
means every module reads the same config the same way, and paths never depend on the
current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root = parent of the ``config/`` directory that holds this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Attribute- and dict-style access over the parsed YAML config.

    Nested mappings are wrapped recursively, so ``config.covariance.windows.ewma_lambda``
    works alongside ``config["covariance"]["windows"]["ewma_lambda"]``.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({list(self._data.keys())})"


def _resolve_path(value: str) -> Path:
    """Resolve a configured path against the project root (absolute paths kept as-is)."""
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_config(path: str | Path | None = None) -> Config:
    """Load and return the project configuration.

    All entries under the top-level ``paths`` mapping are converted to absolute
    :class:`pathlib.Path` objects rooted at the project directory.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # Resolve declared paths to absolute Paths before wrapping.
    if isinstance(raw.get("paths"), dict):
        raw["paths"] = {k: str(_resolve_path(v)) for k, v in raw["paths"].items()}

    return Config(raw)


def ensure_directories(config: Config) -> None:
    """Create every directory declared under ``paths``.

    Entries that look like files (they have a suffix, e.g. ``warehouse.duckdb``) have
    their *parent* created instead of the path itself. Safe to call on every run;
    existing directories are left untouched.
    """
    paths = config.paths.to_dict()
    for value in paths.values():
        p = Path(value)
        target = p.parent if p.suffix else p
        target.mkdir(parents=True, exist_ok=True)
