"""Configuration package — YAML config and its loader.

Import surface:  ``from config.loader import load_config, ensure_directories, Config``
"""

from config.loader import Config, ensure_directories, load_config

__all__ = ["Config", "load_config", "ensure_directories"]
