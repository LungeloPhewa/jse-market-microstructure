# Data

Nothing in `data/real/` or `data/synthetic/` is committed.

* **Real data** (licensed; Refinitiv Tick History). Put the two exports,
  unchanged and still compressed, in `data/real/raw/`:

  ```
  JSE-AGL-GRT-Transactions-Raw-1SEP2013-1OCT2013-N128101748*.csv.gz
  JSE-AGL-GRT-MarketDepth-1SEP2013-1OCT2013-N128189430*.csv.gz
  ```

  Then run `python -m jsemicro verify` to compare them with the published
  checksums, and `make real` to run everything.

* **Synthetic data**: `make synthetic` creates `data/synthetic/` from scratch.

`checksums/` holds the SHA-256 of the files behind each published build.
