# Phase 8 — the client-facing report (gather + render)

from __future__ import annotations

import base64
import logging
from pathlib import Path

import pandas as pd
from jinja2 import Template

logger = logging.getLogger("cov.reports.report")


def _b64(path) -> str | None:
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


def _records(df, n=None):
    if df is None or getattr(df, "empty", True):
        return []
    d = df.head(n) if n else df
    return d.reset_index().to_dict("records") if d.index.name else d.to_dict("records")


def _trend_chart(storage, run_date, out_path) -> Path | None:
    rd = storage.get_covariance("rolling_diagnostics", run_date)
    if rd.empty:
        return None
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rd["date"] = pd.to_datetime(rd["date"])
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(rd["date"], rd["effective_rank"], color="#4C72B0", lw=1.6)
    ax1.set_ylabel("effective independent bets", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(rd["date"], rd["avg_corr"], color="#C44E52", lw=1.1, alpha=0.8)
    ax2.set_ylabel("avg correlation", color="#C44E52")
    ax1.set_title("Diversification over time (90-day rolling)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def gather(config, ctx) -> dict:
    a = ctx.artifacts
    run_date = ctx.started_at.strftime("%Y-%m-%d")
    out_dir = Path(config.paths.reports_out)

    book = a.get("book")
    if book is None or getattr(book, "empty", True):
        book = ctx.storage.get_book()
    data = {"meta": {
        "date": run_date, "base_currency": config.base_currency,
        "book_mv": int(book["market_value_usd"].sum()) if book is not None and not book.empty else None,
        "n_instruments": len(book) if book is not None else None,
    }}

    conc = a.get("concentration")
    if conc:
        recon = a.get("fund_recon")
        cov = (100 * float(recon.loc[recon["looked_through"], "book_weight"].sum())
               if recon is not None and not recon.empty else None)
        data["lookthrough"] = {"metrics": conc.get("metrics", {}), "coverage": cov,
                               "top": _records(conc.get("top_true"), 10)}

    cv = a.get("covariance", {}).get("full")
    if cv:
        data["covariance"] = {
            **cv, "mp_plot": _b64(a.get("mp_plot")),
            "trend_plot": _b64(_trend_chart(ctx.storage, run_date, out_dir / f"trend_{ctx.run_id}.png")),
        }

    st = a.get("structure")
    if st:
        data["structure"] = {
            "agreement": st.get("agreement", {}), "absorption": st.get("absorption_ratio"),
            "factors": _records(st.get("factor_table"), 12),
            "dendrogram": _b64(st.get("dendrogram")), "absorption_plot": _b64(st.get("absorption_plot")),
        }

    fa = a.get("factors")
    if fa:
        data["factors"] = {"exposure": _records(fa.get("exposure")),
                           "naming": _records(fa.get("naming")),
                           "mean_r2": round(float(fa["r2"].mean()), 3) if "r2" in fa else None}

    rk = a.get("risk")
    if rk:
        data["risk"] = {
            "annual_vol": rk.get("annual_vol"), "effective_risk_bets": rk.get("effective_risk_bets"),
            "top": _records(rk.get("contributions"), 10),
            "clusters": _records(rk.get("cluster_risk"), 8),
            "stress": _records(rk.get("stress")),
        }
    return data


def render_report(data: dict, out_path) -> Path:
    html = Template(_TEMPLATE).render(d=data, pct=_pct, num=_num, usd=_usd)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_report(config, ctx) -> str:
    data = gather(config, ctx)
    out = Path(config.paths.reports_out) / "cov_report.html"
    render_report(data, out)
    ctx.artifacts["report"] = str(out)
    sections = [k for k in ("lookthrough", "covariance", "structure", "factors", "risk") if k in data]
    logger.info("Report written -> %s (%d sections: %s)", out, len(sections), ", ".join(sections))
    return f"ok: report with {len(sections)} sections -> {out}"


def _pct(x, dp=2):
    try:
        return f"{float(x) * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(x, dp=2):
    try:
        return f"{float(x):.{dp}f}"
    except (TypeError, ValueError):
        return "—"


def _usd(x):
    try:
        return f"${int(x):,}"
    except (TypeError, ValueError):
        return "—"


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>COV report</title>
<style>
 body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:940px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
 h1{font-size:1.55rem;margin-bottom:.2rem} h2{font-size:1.15rem;margin-top:2.4rem;border-bottom:2px solid #eee;padding-bottom:.3rem}
 .sub{color:#777;margin-bottom:1rem}
 .cards{display:flex;flex-wrap:wrap;gap:.7rem;margin:1rem 0}
 .card{flex:1;min-width:140px;background:#f6f7f9;border-radius:8px;padding:.7rem .9rem}
 .card .v{font-size:1.35rem;font-weight:700;color:#2a4d7a} .card .l{color:#777;font-size:.8rem}
 table{border-collapse:collapse;width:100%;margin:.7rem 0;font-size:.9rem}
 th,td{padding:.35rem .55rem;text-align:left;border-bottom:1px solid #eee}
 th{color:#777;font-weight:600;font-size:.78rem;text-transform:uppercase}
 td.n,th.n{text-align:right;font-variant-numeric:tabular-nums} .up{color:#b5341f;font-weight:600}
 img{max-width:100%;border:1px solid #eee;border-radius:6px;margin:.5rem 0}
 footer{color:#999;font-size:.8rem;margin-top:2.5rem;border-top:1px solid #eee;padding-top:.8rem}
</style></head><body>
<h1>COV — Portfolio Structure &amp; Risk Report</h1>
<div class="sub">
 {% if d.meta.book_mv %}Book {{ usd(d.meta.book_mv) }} &middot; {% endif %}
 {{ d.meta.n_instruments }} instruments &middot; base {{ d.meta.base_currency }} &middot; {{ d.meta.date }}</div>

{% if d.lookthrough %}<h2>1 &nbsp;True vs Declared Exposure</h2>
{% set m = d.lookthrough.metrics %}
<div class="cards">
 {% if d.lookthrough.coverage %}<div class="card"><div class="v">{{ num(d.lookthrough.coverage,1) }}%</div><div class="l">of book looked through</div></div>{% endif %}
 <div class="card"><div class="v">{{ m.effective_names_declared }} &rarr; {{ m.effective_names_true }}</div><div class="l">effective names (declared &rarr; true)</div></div>
 <div class="card"><div class="v">{{ pct(m.top10_declared_weight,1) }} &rarr; {{ pct(m.top10_true_weight,1) }}</div><div class="l">top-10 concentration</div></div>
</div>
<table><tr><th>Company</th><th class="n">Declared</th><th class="n">Via ETFs</th><th class="n">True</th></tr>
{% for r in d.lookthrough.top %}<tr><td>{{ r.name }}</td><td class="n">{{ pct(r.direct_weight) }}</td>
<td class="n">{{ pct(r.indirect_weight) }}</td><td class="n"><b>{{ pct(r.true_weight) }}</b></td></tr>{% endfor %}</table>
{% endif %}

{% if d.covariance %}<h2>2 &nbsp;Correlation Structure</h2>
<div class="cards">
 <div class="card"><div class="v">{{ d.covariance.n_factors }}</div><div class="l">factors above noise floor</div></div>
 <div class="card"><div class="v">{{ pct(d.covariance.market_var_share,0) }}</div><div class="l">variance from the market</div></div>
 <div class="card"><div class="v">{{ num(d.covariance.effective_rank_detoned,0) }}</div><div class="l">effective bets (detoned)</div></div>
 <div class="card"><div class="v">{{ d.covariance.lw_delta }}</div><div class="l">Ledoit-Wolf shrinkage &delta;</div></div>
</div>
{% if d.covariance.mp_plot %}<img src="data:image/png;base64,{{ d.covariance.mp_plot }}" alt="MP spectrum">{% endif %}
{% if d.covariance.trend_plot %}<img src="data:image/png;base64,{{ d.covariance.trend_plot }}" alt="diversification trend">{% endif %}
{% endif %}

{% if d.structure %}<h2>3 &nbsp;Structure — Clusters vs Declared Sectors</h2>
{% set g = d.structure.agreement %}
<div class="cards">
 <div class="card"><div class="v">{{ g.n_clusters }} vs {{ g.n_sectors }}</div><div class="l">clusters vs declared sectors</div></div>
 <div class="card"><div class="v">{{ g.n_divergent }}</div><div class="l">behaviourally mislabeled holdings</div></div>
 <div class="card"><div class="v">{{ num(d.structure.absorption,2) }}</div><div class="l">absorption ratio (top-5)</div></div>
</div>
<table><tr><th class="n">Factor</th><th class="n">Var</th><th>Dominant sector</th><th>Top drivers</th></tr>
{% for r in d.structure.factors %}<tr><td class="n">{{ r.factor }}</td><td class="n">{{ pct(r.variance_share,1) }}</td>
<td>{{ r.dominant_sector }}</td><td>{{ r.drivers }}</td></tr>{% endfor %}</table>
{% if d.structure.dendrogram %}<img src="data:image/png;base64,{{ d.structure.dendrogram }}" alt="dendrogram">{% endif %}
{% if d.structure.absorption_plot %}<img src="data:image/png;base64,{{ d.structure.absorption_plot }}" alt="absorption">{% endif %}
{% endif %}

{% if d.factors %}<h2>4 &nbsp;Economic Factor Exposure</h2>
<p>Factors explain ~{{ pct(d.factors.mean_r2,0) }} of the average holding's variance.</p>
<table><tr><th>Factor</th><th class="n">Net beta</th></tr>
{% for r in d.factors.exposure %}<tr><td>{{ r.factor }}</td><td class="n">{{ num(r.net_beta,3) }}</td></tr>{% endfor %}</table>
{% if d.factors.naming %}<table><tr><th>PCA factor</th><th>Economic name</th><th class="n">R&sup2;</th></tr>
{% for r in d.factors.naming %}<tr><td>{{ r.pca_factor }}</td><td>{{ r.economic_name }}</td><td class="n">{{ r.r2 }}</td></tr>{% endfor %}</table>{% endif %}
{% endif %}

{% if d.risk %}<h2>5 &nbsp;Risk Budget &amp; Stress</h2>
<div class="cards">
 <div class="card"><div class="v">{{ pct(d.risk.annual_vol,1) }}</div><div class="l">annualized volatility</div></div>
 <div class="card"><div class="v">{{ num(d.risk.effective_risk_bets,1) }}</div><div class="l">effective risk bets</div></div>
</div>
<table><tr><th>Holding</th><th class="n">Weight</th><th class="n">% of risk</th></tr>
{% for r in d.risk.top %}<tr><td>{{ r.instrument }}</td><td class="n">{{ pct(r.weight) }}</td><td class="n"><b>{{ pct(r.pct_risk) }}</b></td></tr>{% endfor %}</table>
<h3 style="font-size:1rem;color:#555">Stress — portfolio vol under correlation regimes</h3>
<table><tr><th>Scenario</th><th class="n">Avg corr</th><th class="n">Annual vol</th><th class="n">vs base</th></tr>
{% for r in d.risk.stress %}<tr><td>{{ r.scenario }}</td><td class="n">{{ num(r.avg_corr,2) }}</td>
<td class="n">{{ pct(r.annual_vol,1) }}</td><td class="n">{{ num(r.vs_baseline,2) }}x</td></tr>{% endfor %}</table>
{% endif %}

<footer>Generated by the COV pipeline (Phases 1&ndash;8). Measurement instrument — the covariance
matrix is never inverted for construction. Data local to this machine.</footer>
</body></html>"""
