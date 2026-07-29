# COV — Portfolio Correlation & Exposure Model

A **measurement and transparency instrument** for a multi-custodian liquid portfolio.
It answers one core question:

> **How many independent bets does this portfolio actually contain, and where is it
> secretly over-exposed?**

Given a raw multi-custodian position sheet, COV nets it into a clean book, sees *through*
your ETFs to your true single-name exposure, denoises the correlation matrix with random-
matrix theory, and reports how many real factors you hold, which holdings secretly co-move,
your economic factor exposure, and how the whole thing behaves in a crisis — end to end,
reproducibly, with every matrix carrying its own honesty labels.

> This repository ships a **fictional ~150-instrument, 20-sector example portfolio**
> (`examples/portfolio_example.csv`) so the entire pipeline runs out of the box — every
> ticker is a liquid US listing that prices cleanly, and **25 of its ETFs have live
> holdings sources**, so look-through really bites (it reveals, e.g., that broad S&P
> funds held several times over quietly triple your true NVIDIA/Apple exposure). No real
> data is included. To analyse your own book, see [Using your own data](#using-your-own-data).

### What it deliberately does NOT do
Predict returns · recommend trades · optimise weights · handle privates. It is a
**measurement** tool, not an optimiser — the covariance matrix is *never inverted* for
construction.

---

## Quick start

```bash
# 1. Environment (one time)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Run the full 8-phase pipeline on the example portfolio (fetches market data)
python -m cov.pipeline

# 3. Open the generated report
open reports/output/cov_report.html      # macOS  (xdg-open on Linux)

# 4. Run the tests (fully offline)
pytest -q
```

The run writes a timestamped log to `logs/`, prints a per-phase summary, persists every
intermediate under `data/`, and produces a self-contained HTML report with five sections
and four charts.

---

## The eight phases

Each phase is independently useful and runs in order; each writes its outputs to `data/`
and hands off to the next.

| Phase | Home | What it does | Deliverable |
|------:|------|--------------|-------------|
| 1 | `resolution/` | Net raw positions across custodians into one canonical book | Clean book; total ties out to the cent |
| 2 | `engine/` | Fetch prices, FX→USD, build the returns matrix + quality gates | Returns panel (dates × instruments) |
| 3 | `resolution/` | See through ETFs to effective single-name exposure | "Declared vs. true concentration" |
| 4 | `engine/` | Denoise covariance: Marchenko–Pastur, Ledoit–Wolf, RIE, detoning | Clean matrices + MP diagnostic plot |
| 5 | `analytics/` | PCA factors + clustering vs. the declared sectors; absorption ratio | "Your N sectors are really K factors" |
| 6 | `analytics/` | Regress holdings on market/style/macro factors (FRED) | Economic factor exposure |
| 7 | `analytics/` | Risk budget (MCTR/CCTR), effective bets, correlation stress | Risk decomposition + stress table |
| 8 | `reports/` | Consolidate everything into one HTML report | The client-facing deliverable |

**The methods:** cross-custodian netting · ETF look-through · Marchenko–Pastur noise
floor (random matrix theory) · Ledoit–Wolf shrinkage · rotationally-invariant estimator
(RIE) cleaning · market-mode detoning · EWMA / rolling correlation · Spearman robustness ·
hierarchical clustering on correlation distance √(2(1−ρ)) · absorption ratio · factor
regression · marginal/component risk contribution · historical-episode & tail-correlation
stress.

---

## Using your own data

Your real portfolio and config stay **local** — they are git-ignored and never leave your
machine.

1. Drop your position sheet at `data/raw/custodian/positions_raw.csv` with columns:
   `raw_id, custodian, sector_label, ticker, isin, name, comment, etf_fee, shares,
   cost_basis_usd, market_value_usd` (see `examples/portfolio_example.csv` for the shape).
2. Copy the example configs and edit them for your book (the copies are git-ignored):
   ```bash
   for f in entity_overrides symbol_overrides constituent_overrides \
            etf_classification etf_holdings_sources; do
     cp config/$f.example.yaml config/$f.yaml
   done
   ```
3. In `config/config.yaml` set `ingestion.input_file` to your CSV and repoint the
   `*_file:` entries from `*.example.yaml` to your `*.yaml` copies.
4. `python -m cov.pipeline`.

---

## Storage

Parquet files on disk are the **system of record** (durable, portable, auditable);
DuckDB is the **query engine** over them plus the home for small hand-edited reference
tables. Nothing important lives only in the database — the files are the truth.

```
data/
  raw/          custodian/ holdings/ prices/     # untouched downloads — audit trail (git-ignored)
  parquet/      prices/year=YYYY  holdings/asof=YYYY-MM  factors/   # system of record
  snapshots/    book/run=…  covariance/run=…      # immutable per-run analytical outputs
  warehouse.duckdb                                # query engine + reference tables
```

[`cov/storage.py`](cov/storage.py) is the **only** module that touches disk — everything
else calls `get_returns(...)`, `write_snapshot(...)`, etc. Writes append (never overwrite);
given a `run=YYYY-MM-DD` folder, every downstream number is reproducible. Ops burden: none —
no server, no Docker, no cloud. Backup = copy `data/`.

## Stack

Python 3.13 · numpy / pandas / scipy / scikit-learn / statsmodels · Parquet (pyarrow) +
DuckDB · yfinance (prices) · FRED via pandas-datareader (macro factors) · matplotlib ·
Jinja2 (report). Everything runs locally.

## Working principles
- Clean inputs beat clever math on dirty ones — Phases 1–3 are solid before trusting downstream.
- Point-in-time discipline: never overwrite, always append.
- Every computed matrix carries metadata (window, method, condition number, effective rank).
- Batch, not real-time.
- The deliverable is *the disagreement*: "20 declared sectors" vs. "a handful of real factors".
