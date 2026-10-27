# Tests for the configuration layer

from __future__ import annotations

from pathlib import Path

from config.loader import PROJECT_ROOT, Config, ensure_directories, load_config


def test_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)


def test_base_currency_is_set():
    cfg = load_config()
    assert cfg.base_currency in {"USD", "EUR", "CHF", "GBP"}


def test_paths_are_absolute_and_rooted():
    cfg = load_config()
    for value in cfg.paths.to_dict().values():
        p = Path(value)
        assert p.is_absolute(), f"path not absolute: {p}"
        assert str(p).startswith(str(PROJECT_ROOT)), f"path escaped project root: {p}"


def test_nested_access_works():
    cfg = load_config()
    assert cfg.covariance.windows.ewma_lambda == 0.94
    assert cfg["covariance"]["estimator"] == "ledoit_wolf"


def test_key_parameters_present():
    cfg = load_config()
    assert cfg.prices.quality_gates.outlier_sigma == 4.0
    assert cfg.structure.declared_sector_count == 20


def test_ensure_directories_creates_paths(tmp_path, monkeypatch):
    cfg = load_config()
    for key in cfg.paths.to_dict():
        setattr(cfg.paths, key, str(tmp_path / key))
    cfg.paths._data = {k: str(tmp_path / k) for k in cfg.paths._data}
    ensure_directories(cfg)
    assert (tmp_path / "data_dir").exists()
