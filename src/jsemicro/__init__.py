"""jsemicro -- market microstructure of two JSE equities from tick data.

Subpackages
-----------
ingest          raw Refinitiv (TRTH) exports -> flat, trimmed event tables
pipeline        Part I: merge, external sort, clean and classify (Lee-Ready)
eda             Part I: sanity checks, order-flow and inter-arrival stylised facts
microstructure  Part II: bars, stylised facts, price impact, seasonality,
                order-book resilience and the cost of trading
hawkes          Part III: market phases, order-event identification and a
                10-variate Hawkes model of order flow
synthetic       a limit-order-book simulator that writes raw exports in the
                Refinitiv format, so the whole pipeline runs without the
                licensed data
"""
__version__ = "1.0.0"
