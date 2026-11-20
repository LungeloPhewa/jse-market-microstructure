"""
Trim flattened Transactions data to required columns
========================================================

Reduces the 370-column flattened TX output down to just the identity
columns plus the raw fields needed to compute every derived variable in
the analysis (mid-price, micro-price, mid-price change,
inter-arrival time, normalised volume, VWAP) and the required raw fields
(bid, bid volume, ask, ask volume, transaction price, transaction volume).

WHY trim before merging/Stage 2: the full 370-column file is what caused
the earlier out-of-memory error just to inspect it -- carrying that width
through the merge with Market Depth and into Stage 2's sort/clean step
would make an already resource-constrained machine's job much harder for
no benefit, since ~355 of those columns are never used by anything in
this assignment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Identity columns (always keep) + raw fields needed for every required/
# derived variable identified from the assignment's formula sheet.
KEEP_COLUMNS = [
    "event_id", "ric", "ts_utc", "ts_local", "type", "message_type",
    "BID", "BIDSIZE",       # b_ij, v^b_ij -- mid-price, micro-price
    "ASK", "ASKSIZE",       # a_ij, v^a_ij -- mid-price, micro-price
    "MID_PRICE",            # feed-provided mid-price, for cross-checking our own m_ij = (b+a)/2
    "TRDPRC_1", "TRDVOL_1", # transaction price/volume -- required fields, also feed VWAP
    "ACVOL_1",              # cumulative daily volume
    "PRCTCK_1",             # price-tick direction -- may assist the Lee-Ready tick test
    "BID_TICK_1", "ASK_TICK_1",
    "SEQNUM",               # Stage 2's deterministic sort key
    # Market-state and trade-type fields (Assignment 3, Step 0): PRC_QL2 /
    # PRC_QL3 carry the exchange's trading-phase markers (REP, AUC, TRD, CLS
    # with CLA/STL/OSP/ETR), and OFFBK_VOL the cumulative off-book volume,
    # which identifies negotiated trades reported to the exchange.
    "PRC_QL2", "PRC_QL3", "OFFBK_VOL",
]


def trim_columns(input_path: str | Path, output_path: str | Path, chunksize: int = 100_000) -> None:
    """Stream through the flattened TX file, keeping only KEEP_COLUMNS.

    WHY chunked rather than a single read + column select: the earlier
    _ArrayMemoryError showed that even loading this file's full width
    exhausts available memory -- selecting columns via `usecols` at read
    time (rather than reading everything then dropping columns) means
    pandas never allocates memory for the ~355 unwanted columns in the
    first place.
    """
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    # Only request columns that actually exist in the file -- guards
    # against a KEEP_COLUMNS entry that turns out to be misspelled or
    # absent for this particular security's data.
    available = pd.read_csv(input_path, nrows=0).columns.tolist()
    cols_to_use = [c for c in KEEP_COLUMNS if c in available]
    missing = [c for c in KEEP_COLUMNS if c not in available]
    if missing:
        log.warning("These expected columns were NOT found and will be skipped: %s", missing)

    first_write = True
    total_rows = 0
    for chunk in pd.read_csv(input_path, usecols=cols_to_use, chunksize=chunksize, low_memory=False):
        chunk.to_csv(output_path, mode="a", header=first_write, index=False)
        first_write = False
        total_rows += len(chunk)

    log.info("Trimmed file written: %d rows, %d columns -> %s", total_rows, len(cols_to_use), output_path)


if __name__ == "__main__":
    # Run the whole data preparation from the submission folder with
    #     python scripts/rebuild_data.py
    # which calls this module with the correct input and output paths.
    raise SystemExit("Run `python scripts/rebuild_data.py` from the submission folder (see README.md).")
