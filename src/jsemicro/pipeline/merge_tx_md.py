"""
Merge Transactions and Market Depth into one event-time table
================================================================

Combines the trimmed TX events and the prepared Market Depth events into
a SINGLE table -- every row from both feeds kept (event-time, per your
decision), with top-of-book fields coalesced onto one set of canonical
column names so downstream code (Stage 2, derived_features.py) doesn't
need to know which feed a given row came from.

DESIGN DECISIONS (confirmed)
-------------------------------
1. Event-time, not trade-centric: every TX row and every MD row is kept
   as its own row in the output -- nothing is collapsed or dropped.
2. Market Depth's L1 values take priority for the canonical BID/ASK/
   BIDSIZE/ASKSIZE columns. In practice, per-row "disagreement" doesn't
   arise here (a TX row never carries MD's L1 fields and vice versa) --
   the real question is which feed's update should be treated as
   authoritative when a forward-fill (done later, on the sorted output)
   crosses a moment where BOTH feeds updated at the exact same
   timestamp. Stage 2's existing sort key already resolves this
   correctly: TX rows carry SEQNUM, MD rows don't, and the sort's
   na_position="last" default places MD rows AFTER TX rows at any exact
   timestamp tie -- so a subsequent .ffill() naturally reflects MD's
   value from that instant onward. No change to Stage 2 was needed.

WHY THIS IS STREAMED RATHER THAN A SINGLE pd.concat()
--------------------------------------------------------
The Market Depth table (7.8M rows x 45 columns) is large enough that
loading it fully alongside TX risks repeating the earlier out-of-memory
failures. Since Stage 2 re-sorts the WHOLE merged table into deterministic
event-time order afterward anyway, this merge step doesn't need to
produce pre-sorted output itself -- it only needs every row from both
feeds present in one file with a CONSISTENT schema. So: write TX (small,
loaded once) as the first row group(s), then stream Market Depth in
batches, coalescing and reindexing each batch onto the same fixed column
list before writing it as a further row group -- exactly the same
fixed-schema lesson that fixed the earlier ragged-CSV bug in
flatten_taq.py, applied here to keep every row group's schema identical.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# The full, fixed set of output columns, in a fixed order -- every row
# group written (whether from TX or an MD batch) is reindexed onto
# exactly this list, guaranteeing schema consistency across the whole
# output file regardless of which feed a given batch came from.
IDENTITY_COLS = ["event_id", "ric", "ts_utc", "ts_local", "type", "source"]
CANONICAL_TOB_COLS = ["BID", "ASK", "BIDSIZE", "ASKSIZE"]  # coalesced top-of-book
TX_ONLY_COLS = [
    "message_type", "ACVOL_1", "ASK_TICK_1", "BID_TICK_1",
    "MID_PRICE", "PRCTCK_1", "SEQNUM", "TRDPRC_1", "TRDVOL_1",
]
MD_LEVEL_COLS = [
    f"L{lvl}-{side}{field}"
    for lvl in range(1, 11)
    for side, field in [("Bid", "Price"), ("Bid", "Size"), ("Ask", "Price"), ("Ask", "Size")]
]

FULL_COLUMNS = IDENTITY_COLS + CANONICAL_TOB_COLS + TX_ONLY_COLS + MD_LEVEL_COLS

# Columns that should always be numeric (float) in the final schema, even
# when entirely NaN for a given batch (e.g. TX rows have no L1-L10
# columns at all) -- explicit casting prevents pyarrow from inferring a
# different (e.g. null/object) type for an all-NaN column in one batch
# versus a float type in another, which would otherwise cause a schema
# mismatch between row groups.
NUMERIC_COLUMNS = CANONICAL_TOB_COLS + [
    c for c in TX_ONLY_COLS if c not in ("message_type",)
] + MD_LEVEL_COLS


def _finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto FULL_COLUMNS and enforce consistent dtypes.

    WHY: this is the direct fix for the schema-drift problem that broke
    the original flattened TX CSV -- every batch written to the merged
    output, regardless of source feed, passes through this same
    reindex-and-cast step, so pyarrow never sees two row groups with
    different column sets or incompatible types for the same column
    name.
    """
    df = df.reindex(columns=FULL_COLUMNS)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in IDENTITY_COLS + ["message_type"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _coalesce_top_of_book(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Populate the canonical BID/ASK/BIDSIZE/ASKSIZE columns.

    WHY separate mappings per source: TX rows already use these exact
    column names natively; MD rows instead carry the same concept under
    L1-BidPrice/L1-AskPrice/L1-BidSize/L1-AskSize -- this function is
    what makes both feeds speak the same column-name "language" before
    they're combined.
    """
    df = df.copy()
    df["source"] = source
    if source == "MD":
        df["BID"] = df["L1-BidPrice"]
        df["ASK"] = df["L1-AskPrice"]
        df["BIDSIZE"] = df["L1-BidSize"]
        df["ASKSIZE"] = df["L1-AskSize"]
    # For source == "TX", BID/ASK/BIDSIZE/ASKSIZE are already the correct
    # native columns -- nothing to do.
    return df


def merge_tx_md(
    tx_path: str | Path,
    md_path: str | Path,
    output_path: str | Path,
    md_batch_size: int = 200_000,
    rics: list[str] | None = None,
) -> None:
    """Merge trimmed TX (CSV) and prepared Market Depth (Parquet) into
    one event-time Parquet file with a consistent, coalesced schema.
    """
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None

    try:
        # --- TX: small enough to load in full ---
        log.info("Loading TX from %s", tx_path)
        tx = pd.read_csv(tx_path, low_memory=False)
        # ISO-8601 strings with offsets; parse explicitly so ts_utc is tz-aware UTC.
        tx["ts_utc"] = pd.to_datetime(tx["ts_utc"], utc=True, format="ISO8601", errors="coerce")
        tx["ts_local"] = tx["ts_utc"].dt.tz_convert("Africa/Johannesburg")
        if rics is not None:
            # The Transactions export also carries index records (e.g. .JTOPI).
            tx = tx[tx["ric"].isin(rics)]
        tx = _coalesce_top_of_book(tx, source="TX")
        tx = _finalize_schema(tx)

        table = pa.Table.from_pandas(tx, preserve_index=False)
        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)
        log.info("Wrote %d TX rows as the first row group.", len(tx))

        # --- Market Depth: streamed in batches to bound memory ---
        log.info("Streaming Market Depth from %s", md_path)
        pf = pq.ParquetFile(md_path)
        n_md_rows = 0
        for batch in pf.iter_batches(batch_size=md_batch_size):
            md_chunk = batch.to_pandas()
            if rics is not None:
                md_chunk = md_chunk[md_chunk["ric"].isin(rics)]
            md_chunk = _coalesce_top_of_book(md_chunk, source="MD")
            md_chunk = _finalize_schema(md_chunk)

            table = pa.Table.from_pandas(md_chunk, schema=writer.schema, preserve_index=False)
            writer.write_table(table)
            n_md_rows += len(md_chunk)
            log.info("Wrote %d MD rows so far...", n_md_rows)
    finally:
        # WHY try/finally: guarantees the file handle is released even if
        # something fails partway through (e.g. a bad md_path) -- without
        # this, a mid-run failure leaves the ParquetWriter's file handle
        # open for as long as the Python session lives, causing a
        # PermissionError on any subsequent attempt to delete/recreate
        # output_path, even after fixing whatever caused the original
        # failure.
        if writer is not None:
            writer.close()

    log.info("Done. Merged event-time table written to %s (TX=%d, MD=%d rows).",
              output_path, len(tx), n_md_rows)


