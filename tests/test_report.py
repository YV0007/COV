# Phase 8 tests — the HTML renderer (pure) and its formatting helpers

from __future__ import annotations

from reports import report as rp


def _fixture():
    return {
        "meta": {"date": "2026-07-27", "base_currency": "USD",
                 "book_mv": 50000000, "n_instruments": 250},
        "lookthrough": {
            "coverage": 31.5,
            "metrics": {"effective_names_declared": 68, "effective_names_true": 49,
                        "top10_declared_weight": 0.33, "top10_true_weight": 0.37},
            "top": [{"name": "Apple", "direct_weight": 0.0039, "indirect_weight": 0.0127,
                     "true_weight": 0.0166}],
        },
        "risk": {"annual_vol": 0.149, "effective_risk_bets": 15.1,
                 "top": [{"instrument": "AMD", "weight": 0.064, "pct_risk": 0.171}],
                 "clusters": [], "stress": [
                     {"scenario": "baseline", "avg_corr": 0.15, "annual_vol": 0.149, "vs_baseline": 1.0},
                     {"scenario": "all_corr_to_0.9", "avg_corr": 0.9, "annual_vol": 0.301, "vs_baseline": 2.02}]},
    }


def test_render_includes_present_sections(tmp_path):
    out = rp.render_report(_fixture(), tmp_path / "r.html")
    html = out.read_text()
    assert "COV — Portfolio Structure" in html
    assert "$50,000,000" in html
    assert "True vs Declared Exposure" in html
    assert "Apple" in html and "1.66%" in html
    assert "Risk Budget" in html
    assert "AMD" in html and "17.10%" in html
    assert "2.02x" in html


def test_render_omits_absent_sections(tmp_path):
    out = rp.render_report(_fixture(), tmp_path / "r.html")
    html = out.read_text()
    assert "Correlation Structure" not in html
    assert "Economic Factor Exposure" not in html


def test_format_helpers():
    assert rp._pct(0.0166) == "1.66%"
    assert rp._usd(50000000) == "$50,000,000"
    assert rp._num(0.149, 1) == "0.1"
    assert rp._pct(None) == "—"
