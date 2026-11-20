# Market Microstructure of the JSE from Tick Data

**A reproducible study of Anglo American (AGLJ.J) and Growthpoint (GRTJ.J) on the Johannesburg Stock Exchange: from raw Refinitiv tick exports to trading-cost measurement and a Hawkes-process model of order flow.**

📖 **Read the book:** <https://LungeloPhewa.github.io/jse-market-microstructure/> · synthetic-data build: [/synthetic/](https://LungeloPhewa.github.io/jse-market-microstructure/synthetic/)

[![build-and-publish](https://github.com/LungeloPhewa/jse-market-microstructure/actions/workflows/publish.yml/badge.svg)](https://github.com/LungeloPhewa/jse-market-microstructure/actions/workflows/publish.yml)

---

## What this is

One month (September 2013) of trades, quotes and ten-level order-book depth for two JSE shares, about 2.3 million transaction messages and 7.8 million book updates. The data go through a production-style pipeline and are analysed in three parts:

| Part | Question | Methods |
|---|---|---|
| **I · Data engineering** | How do you turn vendor tick exports into a clean, signed, time-ordered event table on a laptop? | streaming FID pivot with a fixed schema, Parquet, external merge sort ($O(N\log C)$ heap merge), Lee–Ready trade classification, power-law tail MLE (Clauset et al.) |
| **II · Microstructure and trading costs** | What does it cost to trade, how does it depend on size and time of day, and how fast does liquidity come back? | OHLCV/VWAP/micro-price bars, return stylised facts and Hill tail index, price-impact curves, intraday seasonality, order-book resilience, effective/quoted/realised spreads, VPIN, the square-root law |
| **III · Order-flow modelling** | Which orders trigger which, how strongly and how fast? | market-state classification, inference of ten order event types from TAQ data, 10-variate Hawkes process with exponential kernels, MLE with analytic gradients (numba), generalised-residual diagnostics |

## Selected findings

| | AGLJ.J | GRTJ.J |
|---|---|---|
| Effective spread | 5.6 bps | 10.8 bps |
| …of which lost to adverse selection within 60 s | 74% | 55% |
| Size exponent of trade cost (square-root law: 0.5) | 0.06 | 0.14 |
| Lag-1 autocorrelation of trade signs | 0.26 | 0.31 |
| Book events per day | ≈ 440,000 | ≈ 17,000 |
| Hawkes branching matrix, spectral radius (median day) | 0.84 | 0.83 |
| Ask-side refill after a buy market order: half-life | 56 ms | 54 ms |

* **The relative tick sets liquidity.** Both shares have the same 1c tick, but at a tenth of the price GRTJ.J pays about twice as much to trade.
* **When you trade matters more than how much.** Single-trade cost barely rises with size (orders are split), but the first ten minutes have 1.7–2.7× the average volatility and wider spreads.
* **Order flow is mostly endogenous.** A ten-variate Hawkes model calibrated on 42 stock-days finds near-critical self-excitation: quotes are pulled within milliseconds of a market order and refilled within a fraction of a second.
* **Testing on known answers matters.** Running the full pipeline on simulated data with known parameters exposed a silent pandas-3 time-unit bug that had widened two event-matching windows a thousandfold. It is now fixed and has a regression test.

## Reproducibility

```bash
git clone https://github.com/LungeloPhewa/jse-market-microstructure.git
cd jse-market-microstructure
make docker-build
make docker-synthetic     # tests → simulate exports → full pipeline → book, about 5 minutes
```

* **The data are licensed** (Refinitiv/LSEG) and are not in this repository. A built-in **synthetic market** (a Hawkes-driven limit order book) writes exports in *exactly* the vendor's format, so the entire pipeline runs and is tested without them. CI does this on every push and publishes the result under `/synthetic/`.
* **The real-data results** are published from Quarto's `_freeze/` cache. **SHA-256 checksums** of the licensed inputs are committed, so a licence holder can verify that they have the same files (`python -m jsemicro verify`) and reproduce everything with `make real`.
* **Snakemake** declares the pipeline. **`requirements-lock.txt`** pins every package. **Docker** pins Python and Quarto. Fixed seeds make the synthetic data byte-identical on every run. **pytest** checks the building blocks against known answers, including parameter recovery for the Hawkes estimator.

## Repository layout

| Path | Contents |
|---|---|
| `src/jsemicro/` | the Python package: `ingest`, `pipeline`, `eda` (Part I), `microstructure` (Part II), `hawkes` (Part III), `synthetic`, `config`, `integrity` |
| `workflow/Snakefile` | the pipeline, raw exports → cached results |
| `tests/` | unit tests |
| `index.qmd`, `chapters/` | the Quarto book (reads cached results; shows the code that produced them) |
| `_freeze/` | executed real-data chapters |
| `data/checksums/` | SHA-256 of the inputs behind each build |
| `Dockerfile`, `Makefile`, `requirements-lock.txt` | the environment and one command per stage |
| `docs/` | data provenance notes and the development plan |

Run `make help` for every command. Without Docker you need Python 3.13 and Quarto 1.8: `pip install -r requirements-lock.txt && pip install -e .`, then `make synthetic`.

## Author

**Lungelo Phewa**, Statistical Sciences, University of Cape Town. The three parts began as assignments for a postgraduate course in high-frequency trading and market microstructure and were consolidated, corrected and extended for this repository; see the book's introduction and its authorship appendix.

Code: MIT licence. Market data: © LSEG/Refinitiv, not redistributed.
