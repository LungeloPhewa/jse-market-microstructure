# Development and commit plan

One focused task per day, each ending in a commit (and push) that leaves the repository working. Make each commit on the day you actually do and check the task. Do not back-date commits: an honest history is part of reproducibility.

Conventional prefixes: `feat:` new analysis, `fix:` correction, `docs:` writing, `test:`, `build:` environment, `ci:`.

## Setting up once

```bash
git init -b main
git remote add origin git@github.com:LungeloPhewa/jse-market-microstructure.git
```

On GitHub: **Settings → Pages → Source: GitHub Actions**.

Never commit `data/real/`; `.gitignore` already excludes it, along with all `*.csv.gz`, `*.parquet` and `*.pkl` files. Check `git status` before every commit.

## Day by day

| Day | Task | `git add` | Commit message |
|---|---|---|---|
| 1 | Skeleton, licence, ignore rules | `.gitignore .dockerignore LICENSE pyproject.toml README.md src/jsemicro/__init__.py src/jsemicro/config.py` | `build: package skeleton, licence and path configuration` |
| 2 | Ingestion: FID flattening and trimming | `src/jsemicro/ingest/flatten_taq.py src/jsemicro/ingest/trim_tx_columns.py src/jsemicro/ingest/__init__.py` | `feat: stream and flatten the TRTH Transactions export` |
| 3 | Depth feed and auction prices | `src/jsemicro/ingest/market_depth_prepare.py src/jsemicro/ingest/auction_prices.py` | `feat: depth feed to Parquet; daily auction prices` |
| 4 | Ingestion tests | `tests/conftest.py tests/test_ingest.py` | `test: FID flattening, GMT handling, auction prints` |
| 5 | Merge and external sort | `src/jsemicro/pipeline/__init__.py src/jsemicro/pipeline/merge_tx_md.py src/jsemicro/pipeline/taq_stage2_ordering_and_cleaning.py src/jsemicro/pipeline/taq_stage3_merge.py` | `feat: event-time merge and external merge sort` |
| 6 | Clean, compact, classify | `src/jsemicro/pipeline/stage4_clean_classify.py src/jsemicro/pipeline/derived_features.py tests/test_pipeline.py` | `feat: cleaning and Lee-Ready classification, with tests` |
| 7 | Part I EDA | `src/jsemicro/eda/` | `feat: sanity checks, order-flow ACF, inter-arrival power law` |
| 8 | Part II data and bars | `src/jsemicro/microstructure/__init__.py src/jsemicro/microstructure/data_io.py src/jsemicro/microstructure/book.py src/jsemicro/microstructure/bar_construction.py src/jsemicro/microstructure/stylised_facts.py` | `feat: quote/trade loaders, OHLCV/VWAP/micro-price bars, stylised facts` |
| 9 | Impact, seasonality, resilience | `src/jsemicro/microstructure/price_impact.py src/jsemicro/microstructure/intraday_seasonality.py src/jsemicro/microstructure/order_book_resilience.py` | `feat: price impact, intraday seasonality and book resilience` |
| 10 | Cost of trading | `src/jsemicro/microstructure/cost_of_trading.py` | `feat: spreads, VPIN, expected shortfall and square-root law` |
| 11 | Part III phases and events | `src/jsemicro/hawkes/__init__.py src/jsemicro/hawkes/market_phases.py src/jsemicro/hawkes/a3_data.py src/jsemicro/hawkes/order_events.py` | `feat: market-state markers and ten order event types` |
| 12 | Hawkes model | `src/jsemicro/hawkes/hawkes.py src/jsemicro/hawkes/a3_model.py src/jsemicro/hawkes/hawkes_plots.py src/jsemicro/hawkes/a3_report.py src/jsemicro/hawkes/run.py` | `feat: 10-variate Hawkes MLE, residual tests and bubble plots` |
| 13 | Synthetic market | `src/jsemicro/synthetic/` | `feat: Hawkes-driven LOB simulator writing TRTH-format exports` |
| 14 | Time-unit fix and its regression test | `src/jsemicro/hawkes/order_events.py tests/test_statistics.py` | `fix: unit-safe ns conversion in event identification (pandas 3 stores us)` |
| 15 | Pipeline, CLI and integrity | `workflow/Snakefile src/jsemicro/__main__.py src/jsemicro/integrity.py data/checksums/` | `build: Snakemake pipeline, CLI and SHA-256 checksums` |
| 16 | Environment | `requirements-lock.txt Dockerfile Makefile` | `build: locked dependencies, Docker image and Makefile` |
| 17 | Book skeleton and Part I chapters | `_quarto.yml styles.css _quarto-synthetic.yml _environment-synthetic references.bib index.qmd chapters/01-introduction.qmd chapters/02-pipeline.qmd chapters/03-order-flow.qmd chapters/_synthetic-banner.qmd chapters/assets/ src/jsemicro/reporting.py` | `docs: book configuration, introduction and Part I` |
| 18 | Part II chapters | `chapters/_part2-setup.qmd chapters/04-bars-stylised-facts.qmd chapters/05-impact-seasonality-resilience.qmd chapters/06-cost-of-trading.qmd` | `docs: Part II, microstructure and trading costs` |
| 19 | Part III chapters | `chapters/_part3-setup.qmd chapters/07-market-phases-events.qmd chapters/08-hawkes.qmd` | `docs: Part III, Hawkes model of order flow` |
| 20 | Reproducibility, conclusion, appendices | `chapters/09-reproducibility.qmd chapters/10-conclusion.qmd chapters/references.qmd chapters/A-synthetic-data.qmd chapters/B-code-reference.qmd chapters/C-ai-statement.qmd` | `docs: reproducibility chapter, conclusion and appendices` |
| 21 | Real-data render (frozen results) | `_freeze/` | `docs: freeze real-data results` |
| 22 | CI and publishing | `.github/workflows/publish.yml docs/commit-plan.md` | `ci: build image, reproduce on synthetic data, publish to Pages` |
| 23 | Release | any fixes | `fix: ...`, then `git tag -a v1.0.0 -m "First public version"` and `git push --tags` |

The `diagnostics/` scripts (development diagnostics kept for transparency) can go in with day 7. The `tools/` helpers used for porting the reports can go in with day 17.

## Every day

```bash
make test                 # never commit with failing tests
git status                # no data, no _book/, no *.parquet staged
git add <the day's files>
git commit -m "<message>"
git push
```

## Day 21 in detail: freezing the real results

With the licensed exports in `data/real/raw/`:

```bash
python -m jsemicro verify     # same files as the published results?
make real                     # pipeline + render; writes _freeze/
git add _freeze && git commit -m "docs: freeze real-data results" && git push
```

`_freeze/` holds the executed output of every chapter: figures, tables and inline numbers, but no raw data. CI publishes the real-data book from it without needing the data. Whenever a chapter's source changes, rerun `make render` locally and commit the updated `_freeze/`.
