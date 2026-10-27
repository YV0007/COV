# COV — portfolio correlation & exposure engine

Takes one raw multi-custodian position sheet. Returns how many independent bets the book
actually holds, which positions co-move, where ETF overlap concentrates single-name risk,
and what the portfolio vol does when correlations go to 0.9.

Measurement only: no return forecasts, no position sizing, no weight optimisation. The
covariance matrix is never inverted.

Ships with a fictional example book (`examples/portfolio_example.csv`, 189 rows, 20
sectors, 25 ETFs with live holdings URLs) so the pipeline runs end to end without private
data.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m cov.pipeline                   # full 8 phases, fetches market data
open reports/output/cov_report.html
pytest -q                                # offline
```

---

## Approach

Six decisions, in dependency order. Everything downstream inherits them.

**1. Resolve identity before measuring anything.** Positions group by ISIN first, ticker
second, hand-maintained override table third (`config/entity_overrides.yaml`), with an
explicit keep-separate list and a collision guard. Netting across custodians is pure
summation, so Phase 1 signs off on an invariant, not a tolerance: parsed total ==
netted total, to the cent.

**2. Decompose funds before correlating them.** An ETF's return series is not its risk.
Two S&P trackers plus a Nasdaq fund plus a semis fund is one position in NVDA held four
times. Phase 3 pulls issuer constituent CSVs, resolves each constituent to a canonical id
(ISIN → ticker → name slug → override), and multiplies through. Declared weights and true
weights are reported side by side.

**3. Convert to one currency before returns, not after.** Prices go to USD at spot, then
log returns. FX co-movement then shows up as the shared factor a USD holder actually
bears, instead of being averaged away.

**4. Hold coin exposure as spot, not as the wrapper.** `prices.crypto_spot` maps crypto
ETF positions to `BTC-USD` / `ETH-USD` / `SOL-USD`. Long clean history instead of a short
post-launch stub. Equities with crypto beta (COIN) stay equities.

**5. Treat the sample correlation matrix as mostly noise.** For N instruments over T days,
q = N/T. Marchenko–Pastur puts the noise edge at λ₊ = (1+√q)²; eigenvalues below it carry
no information about the next period. Eigenvalues are therefore shrunk (Ledoit–Wolf), then
replaced (RIE), then the market mode is stripped, before any clustering or risk number is
produced.

**6. Every matrix carries its own metadata.** Window, estimator, shrinkage intensity δ, q,
λ₊, factor count above the noise edge, condition number, effective rank, coverage %.
Written to `snapshots/covariance/` and to the `run_metadata` table on every run. A number
with no window and no condition number is not usable.

---

## Pipeline

Eight phases, each independently useful, each writing its outputs to disk and handing
artifacts forward through a shared `RunContext`.

| # | Module | In → out | Core method |
|---|--------|----------|-------------|
| 1 | `resolution/ingest, entity, netting` | raw CSV → netted book | ISIN/ticker/override resolution, custodian netting, cent-level reconciliation |
| 2 | `engine/fetch, returns, quality` | book → returns panel (dates × instruments) | yfinance pull (threaded), spot FX to USD, log returns, 4 quality gates |
| 3 | `resolution/lookthrough` | book + issuer CSVs → effective exposures | constituent resolution, weight multiplication, re-sum invariant |
| 4 | `engine/covariance, windows` | returns → cleaned correlation matrices | MP edge, Ledoit–Wolf, RIE, detoning, EWMA, Spearman cross-check |
| 5 | `analytics/structure` | matrices → factor & cluster structure | PCA loadings, Ward clustering on d=√(2(1−ρ)), absorption ratio |
| 6 | `analytics/factors` | returns + FRED/market series → betas | OLS per holding, exposure aggregation, PCA-factor naming |
| 7 | `analytics/stress` | matrices + weights → risk budget | MCTR/CCTR, effective bets, episode replay, correlation shock |
| 8 | `reports/report` | all artifacts → one HTML file | Jinja2, base64-inlined charts, 5 sections |

Phases are a list of `(number, name, callable)` in `cov/pipeline.py`. Each callable is
`(config, ctx) -> status str`, wrapped so a failure logs a full traceback and the run
continues. Adding or reordering a phase is an edit to that list.

---

## Engine internals

### Phase 4 — covariance (`engine/covariance.py`)

Block preparation drops rows below 50% cross-sectional coverage, then columns below 90%
time coverage, then truncates to the first dense row. Remaining gaps are zero-filled, so
N and T are explicit rather than implied.

| Step | Formula / call | Purpose |
|------|----------------|---------|
| MP edges | q = N/T, λ± = (1±√q)² | noise floor; factors = #{λ > λ₊} |
| Shrinkage | `LedoitWolf()`, δ reported | pull sample eigenvalues toward the mean |
| RIE | ξ_k = λ_k / \|1 − q + q·λ_k·s(λ_k − iη)\|², s(z) = (1/N)Σ 1/(z − λ_j), η = N^(−1/2) | rotationally-invariant eigenvalue replacement (Bun–Bouchaud–Potters); large modes barely move, the bulk compresses to its centre |
| Clip (alternative) | λ < λ₊ → mean(λ_noise) | Laloux clipping, crude and robust |
| Detone | zero the top eigenmode, renormalise | removes the market tide so sector/style structure surfaces |
| Metadata | eff. rank = (Σλ)²/Σλ², cond = λ_max/λ_min | effective rank ≈ number of independent bets |

Trace is rescaled after every eigenvalue operation, so total variance is preserved.
Two windows are produced per run: full history, and EWMA with λ=0.94 (T_eff =
(1+λ)/(1−λ) ≈ 32 for the MP edge). Rolling 90-day diagnostics step every 21 days.
Spearman correlation runs as a cross-check and flags divergent pairs.

### Phase 5 — structure (`analytics/structure.py`)

PCA loadings on the detoned matrix give each eigenvector its top-k contributing holdings.
Ward linkage on correlation distance d = √(2(1−ρ)) produces k clusters, cross-tabulated
against the declared sector labels: the output is the disagreement, e.g. 20 declared
sectors resolving into a handful of statistical clusters. Absorption ratio = share of
variance in the top 5 eigenvectors, tracked rolling as a crowding signal.

### Phase 6 — factor overlay (`analytics/factors.py`)

Factor panel = market proxy + style sleeves + FRED macro (DGS10, HY OAS, trade-weighted
USD, WTI) + crypto beta, behind a `FactorFetcher` ABC. OLS per holding gives betas and R²;
portfolio exposure is the weight-dot-beta aggregate. PCA factor returns are then regressed
on the named factors, which is what turns "eigenvector 3" into an economic label.

### Phase 7 — risk & stress (`analytics/stress.py`)

Covariance rebuilt from the cleaned correlation and realised vols. MCTR = (Σw)/σ_p, CCTR =
w ⊙ MCTR, pct_risk = CCTR/σ_p; effective risk bets = 1/Σ(pct_risk²). Risk is then
aggregated by cluster and compared against weight (`risk_vs_weight`). Stress table reprices
portfolio vol under: baseline, each historical episode (COVID 2020, 2022 rate shock —
correlations recomputed inside the window, baseline-filled where data is thin), and a
forced all-off-diagonal ρ = 0.9.

---

## Architecture

Dataflow. Each box persists to disk before the next reads it.

```
examples/portfolio_example.csv  (or data/raw/custodian/positions_raw.csv)
          │
          │  P1  ingest → entity resolution → netting → reconcile
          ▼
    netted book ─────────────────────────► snapshots/book/run=DATE
          │                                        │
   P2     │ yfinance + FX + log returns     P3     │ issuer holdings CSVs
          ▼                                        ▼
    returns panel ──► parquet/prices/year=     effective exposures
          │                                        │  (true single-name weights)
   P4     │ LW shrinkage → RIE → detone            │
          ▼                                        │
  cleaned + detoned correlation ◄──────────────────┘  weights feed P7
          │
          ├─ P5 PCA / Ward clusters / absorption
          ├─ P6 factor betas (FRED, market, style)
          └─ P7 MCTR, effective bets, stress
                      │
                      ▼  P8
            reports/output/cov_report.html
```

Three rules hold the layering together:

1. **`cov/storage.py` is the only module that touches disk.** Everything else calls
   `get_returns()`, `write_snapshot()`, `get_entity_map()`. Swapping the storage engine or
   data source does not touch the engine.
2. **External data sits behind an ABC.** `PriceFetcher`, `HoldingsFetcher`,
   `FactorFetcher`. Tests inject fakes, which is why `pytest -q` is fully offline.
3. **Writes append, never overwrite.** New dated file or partition each time. Given a
   `run=YYYY-MM-DD` folder, every downstream number is reproducible.

### Storage

Parquet on disk is the system of record. DuckDB is the SQL layer over it plus the home for
small hand-edited reference tables. Nothing important lives only in the database.

```
data/
  raw/                custodian/  holdings/  prices/        # untouched downloads, audit trail
  parquet/            prices/year=YYYY/                     # system of record
                      holdings/asof=YYYY-MM/
                      factors/
  snapshots/          book/run=YYYY-MM-DD/                  # immutable per-run outputs
                      covariance/run=YYYY-MM-DD/
                      exposure/run=YYYY-MM-DD/
  warehouse.duckdb                                          # query engine + reference tables
```

| Data | Store | Why |
|------|-------|-----|
| Daily prices / returns | Parquet, partitioned by year | bulk columnar read into matrices |
| ETF holdings snapshots | Parquet, partitioned by as-of month | versioned, append-only |
| Factor / FX series | Parquet | read in bulk |
| Book, matrices, exposures per run | Parquet under `snapshots/` | immutable, point-in-time |
| Entity map, instrument metadata | DuckDB table | small, hand-edited, joined constantly |
| Run metadata, data-quality flags | DuckDB table | written per run, queried across runs |

Rule of thumb: bulk numeric time-series → Parquet; small hand-edited or constantly joined
things → DuckDB. Reads go through DuckDB even for Parquet, so a returns panel can be joined
to the entity map in one statement.

Ops: none. No server, no Docker, no cloud. Backup is `cp -r data/`.

### Repo layout

```
cov/          pipeline orchestrator, storage layer, logging
resolution/   ingest, entity resolution, netting, ETF look-through     (P1, P3)
engine/       price fetch, FX, returns, quality gates, covariance      (P2, P4)
analytics/    structure, factor overlay, risk & stress                 (P5-P7)
reports/      Jinja2 template + report assembly                        (P8)
config/       config.yaml + typed loader + override tables
tests/        one module per phase, offline via injected fakes
examples/     fictional example portfolio
```

---

## Configuration

Single file, `config/config.yaml`, read through a typed loader. The parameters worth
knowing:

| Key | Default | Effect |
|-----|---------|--------|
| `base_currency` | USD | conversion target before any correlation math |
| `prices.start_date` | 2016-01-01 | history window; short names are flagged, not dropped |
| `prices.quality_gates` | 5% missing, 5 stale days, 4σ | flag thresholds, per instrument |
| `covariance.estimator` | `ledoit_wolf` | `ledoit_wolf` or sample |
| `covariance.clean` | `rie` | `rie`, `clip`, or `none` |
| `covariance.detone` | true | strip the market eigenmode |
| `covariance.windows` | 90d rolling, EWMA λ=0.94 | diagnostics and second matrix |
| `structure.clustering_linkage` | ward | linkage on d=√(2(1−ρ)) |
| `structure.absorption_ratio_top_n` | 5 | eigenvectors counted as systemic |
| `stress.scenarios.all_correlations_to` | 0.9 | forced off-diagonal shock |
| `stress.episodes` | COVID 2020, rate shock 2022 | replay windows |

## Using your own data

Real portfolio and configs stay local: both are git-ignored.

```bash
# 1. position sheet -> data/raw/custodian/positions_raw.csv
#    columns: raw_id, custodian, sector_label, ticker, isin, name, comment,
#             etf_fee, shares, cost_basis_usd, market_value_usd

# 2. private copies of the override tables
for f in entity_overrides symbol_overrides constituent_overrides \
         etf_classification etf_holdings_sources; do
  cp config/$f.example.yaml config/$f.yaml
done

# 3. in config/config.yaml: point ingestion.input_file at your CSV and
#    repoint every *_file: entry from *.example.yaml to your *.yaml copy

python -m cov.pipeline
```

## Stack

Python ≥3.11 · numpy, pandas, scipy, scikit-learn, statsmodels · pyarrow + DuckDB ·
yfinance · FRED via pandas-datareader · matplotlib · Jinja2. Runs locally, batch, offline
except for the data pull.
