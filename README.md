# COV — portfolio correlation & exposure engine

Takes a raw multi-custodian position sheet. Returns how many independent bets the book
holds, which positions co-move, where ETF overlap concentrates single-name risk, and what
portfolio vol does when correlations go to 0.9. Measurement only: no forecasts, no sizing,
no optimisation, and the covariance matrix is never inverted.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m cov.pipeline                  # 8 phases, fetches market data
open reports/output/cov_report.html
pytest -q                               # 82 tests, offline
```

Runs out of the box on a fictional example book (`examples/portfolio_example.csv`: 189
rows, 20 sectors, 25 ETFs with live holdings sources).

## Approach

- **Resolve identity first.** ISIN, then ticker, then override table. Netting is pure
  summation, so Phase 1 asserts an invariant: parsed total == netted total, to the cent.
- **Decompose funds before correlating them.** An ETF's return series is not its risk: two
  S&P trackers plus a Nasdaq fund plus a semis fund is one NVDA position held four times.
- **USD before returns**, so FX co-movement stays a factor instead of being averaged away.
- **Coin exposure as spot, not the wrapper** (`BTC-USD`, `ETH-USD`): long clean history
  instead of a post-launch stub.
- **The sample correlation matrix is mostly noise.** q = N/T; Marchenko–Pastur puts the
  noise edge at λ₊ = (1+√q)². Eigenvalues below it are shrunk, replaced and detoned before
  any clustering or risk number.
- **Every matrix carries its metadata**: window, estimator, δ, q, λ₊, factor count,
  condition number, effective rank, coverage.

## Pipeline

A `(number, name, callable)` list in `cov/pipeline.py`; each callable is
`(config, ctx) -> status str`, wrapped so a failure logs a traceback and the run continues.

| # | Module | In → out | Core method |
|---|--------|----------|-------------|
| 1 | `resolution/ingest, entity, netting` | raw CSV → netted book | ISIN/ticker/override resolution, netting, cent reconciliation |
| 2 | `engine/fetch, returns, quality` | book → returns panel | threaded yfinance, spot FX to USD, log returns, 4 quality gates |
| 3 | `resolution/lookthrough` | book + issuer CSVs → effective exposures | constituent resolution, weight multiplication, re-sum invariant |
| 4 | `engine/covariance, windows` | returns → cleaned matrices | MP edge, Ledoit–Wolf, RIE, detoning, EWMA, Spearman check |
| 5 | `analytics/structure` | matrices → factors & clusters | PCA loadings, Ward on d=√(2(1−ρ)), absorption ratio |
| 6 | `analytics/factors` | returns + FRED → betas | OLS per holding, exposure aggregation, PCA-factor naming |
| 7 | `analytics/stress` | matrices + weights → risk budget | MCTR/CCTR, effective bets, episode replay, ρ shock |
| 8 | `reports/report` | artifacts → HTML | Jinja2, inlined charts, 5 sections |

## Engine internals

| Step | Formula | Purpose |
|------|---------|---------|
| MP edges | q = N/T, λ± = (1±√q)² | noise floor; factors = #{λ > λ₊} |
| Shrinkage | `LedoitWolf()`, δ reported | pull sample eigenvalues toward the mean |
| RIE | ξ_k = λ_k / \|1 − q + q·λ_k·s(λ_k − iη)\|², s(z) = (1/N)Σ 1/(z − λ_j), η = N^(−1/2) | rotationally-invariant eigenvalue replacement (Bun–Bouchaud–Potters); large modes barely move, the bulk compresses to its centre |
| Clip (alt) | λ < λ₊ → mean(λ_noise) | Laloux clipping, crude and robust |
| Detone | zero the top eigenmode, renormalise | strips the market tide so sector structure surfaces |
| Metadata | eff. rank = (Σλ)²/Σλ², cond = λ_max/λ_min | effective rank ≈ independent bets |
| Risk | MCTR = (Σw)/σ_p, CCTR = w ⊙ MCTR, bets = 1/Σ(pct_risk²) | contribution per name, aggregated by cluster |

Block prep drops rows under 50% cross-sectional coverage, then columns under 90% time
coverage, so N and T are explicit, and trace is rescaled after every eigenvalue operation.
Two windows per run: full history, and EWMA λ=0.94 (T_eff = (1+λ)/(1−λ) ≈ 32 for its MP
edge). Phase 5 cross-tabs Ward clusters against declared sectors; the output is the
disagreement. Phase 7 reprices vol under baseline, historical episodes, and forced ρ = 0.9.

## Architecture

```
positions_raw.csv
      │ P1  ingest → resolve → net → reconcile
      ▼
 netted book ────────────────────► snapshots/book/run=DATE
      │                                   │
  P2  │ yfinance + FX + log returns   P3  │ issuer holdings CSVs
      ▼                                   ▼
 returns panel ──► parquet/prices/    effective exposures
      │                                   │
  P4  │ LW shrinkage → RIE → detone       │ weights
      ▼                                   │
 cleaned + detoned correlation ◄──────────┘
      │
      ├─ P5 PCA / clusters / absorption
      ├─ P6 factor betas
      └─ P7 risk budget + stress ──► P8 cov_report.html
```

1. `cov/storage.py` is the only module that touches disk; everything else calls
   `get_returns()`, `write_snapshot()`, `get_entity_map()`.
2. External data sits behind an ABC (`PriceFetcher`, `HoldingsFetcher`, `FactorFetcher`),
   which is why the tests are offline.
3. Writes append, never overwrite, so any `run=YYYY-MM-DD` folder is reproducible.

Parquet is the system of record, partitioned `year=` / `asof=` / `run=`; DuckDB is the SQL
layer over it plus the home for small hand-edited tables. No server, no Docker.

## Configuration

All parameters live in `config/config.yaml` behind a typed loader. The ones that change
results: `covariance.clean` (`rie` | `clip` | `none`), `covariance.detone`,
`covariance.windows`, `structure.absorption_ratio_top_n`, `stress.episodes`,
`prices.start_date`, `prices.quality_gates`.

## Using your own data

Portfolio and configs are git-ignored.

```bash
# sheet -> data/raw/custodian/positions_raw.csv (see examples/ for the columns)
for f in entity_overrides symbol_overrides constituent_overrides \
         etf_classification etf_holdings_sources; do
  cp config/$f.example.yaml config/$f.yaml
done
# point ingestion.input_file at your CSV, repoint every *_file: to your copy
python -m cov.pipeline
```

Python ≥3.11 · numpy, pandas, scipy, scikit-learn, statsmodels · pyarrow + DuckDB ·
yfinance · FRED via pandas-datareader · matplotlib · Jinja2. Local, batch.
