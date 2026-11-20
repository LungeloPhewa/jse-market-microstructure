"""
Market Depth Preparation -- no FID-flattening needed
======================================================

Unlike the Transactions feed, the Market Depth export is confirmed (from
real sample data) to arrive ALREADY WIDE: one row per snapshot, with
columns #RIC, Date[G], Time[G], GMT Offset, Type ("Market Depth"), and
L1..Ln BidPrice/BidSize/AskPrice/AskSize directly -- no FID Name/FID Value
continuation rows anywhere in the file.

WHY THIS IS A SEPARATE, SIMPLER SCRIPT
-----------------------------------------
Because there's no header/continuation row structure to reassemble here,
none of flatten_taq.py's chunk-stitching, event-grouping, or FID-pivoting
logic is needed. This script only needs to: build the same ts_utc/ts_local
timestamp pair, normalise the RIC, assign a persistent event_id, and
tolerantly coerce the L*-Price/L*-Size columns to numeric -- so that its
output lands in the SAME shape flatten_taq.py produces, and both feeds can
be fed into taq_stage2_ordering_and_cleaning.py unmodified.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from zoneinfo import ZoneInfo

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

CHUNK_SIZE = 200_000

RIC_COL = "#RIC"
DATE_COL = "Date[G]"
TIME_COL = "Time[G]"
OFFSET_COL = "GMT Offset"
TYPE_COL = "Type"

JSE_TZ = ZoneInfo("Africa/Johannesburg")
JSE_FIXED_OFFSET_HOURS = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Shared helpers (identical logic to flatten_taq.py, kept in sync deliberately
# so both feeds produce ts_utc/ts_local/event_id the same way)
# ----------------------------------------------------------------------------

def normalize_ric(raw_ric: str) -> str:
    return raw_ric.strip().upper() if isinstance(raw_ric, str) and raw_ric else raw_ric


def parse_timestamps_vectorized(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Vectorised version of flatten_taq.py's parse_timestamps().

    WHY vectorised here (unlike the row-by-row version in flatten_taq.py):
    since there's no row-grouping step for this feed, we can process the
    whole chunk's timestamps at once with pandas' vectorised datetime
    parsing -- meaningfully faster than a Python-level loop, and safe to
    do because every row is already a complete, independent event (no
    header/FID rows to reassemble first).
    """
    combined = df[DATE_COL] + " " + df[TIME_COL]

    # Confirmed from real sample data: Time[G] is inconsistently formatted
    # -- some rows include fractional seconds, others don't. Parse with
    # the more specific format first, then fill in any remaining
    # unparsed rows using the simpler format, then a flexible fallback.
    naive = pd.to_datetime(combined, format="%d-%b-%Y %H:%M:%S.%f", errors="coerce")
    still_missing = naive.isna()
    if still_missing.any():
        naive.loc[still_missing] = pd.to_datetime(
            combined.loc[still_missing], format="%d-%b-%Y %H:%M:%S", errors="coerce"
        )
    still_missing = naive.isna()
    if still_missing.any():
        naive.loc[still_missing] = pd.to_datetime(combined.loc[still_missing], errors="coerce")

    # Date[G] / Time[G] are GMT (the "[G]"): they ARE the UTC instant.
    # "GMT Offset" (+2) only says how far the exchange's local clock is AHEAD
    # of GMT, i.e. local = GMT + offset. The previous version subtracted the
    # offset, treating GMT as if it were local time, which made every
    # timestamp 2 hours early (09:00 SAST became 07:00 SAST) and caused
    # data_io's 09:10-16:50 SAST session filter to drop the real 09:00-11:10
    # and keep the closing auction and after-hours prints instead.
    ts_utc = naive.dt.tz_localize("UTC")
    ts_local = ts_utc.dt.tz_convert(JSE_TZ)

    # Sanity check against the stated offset: local - GMT must equal it.
    _stated = pd.to_numeric(df[OFFSET_COL], errors="coerce").dropna()
    if len(_stated) and not (_stated == JSE_FIXED_OFFSET_HOURS).all():
        raise ValueError(f"Unexpected GMT offsets in the file: {sorted(_stated.unique())}")
    return ts_utc, ts_local


def make_event_id_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorised, persistent event id, disambiguating same-(RIC, timestamp)
    rows in encounter order -- same rationale as flatten_taq.py's
    make_event_id(), just computed for a whole column at once.
    """
    key_cols = df["ric"].astype(str) + "|" + df[DATE_COL] + "|" + df[TIME_COL]
    seq = key_cols.groupby(key_cols).cumcount()
    raw = key_cols + "|" + seq.astype(str)
    return raw.apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:16])


def coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def prepare_market_depth_file(
    input_path: str | Path,
    output_path: str | Path,
    chunksize: int = CHUNK_SIZE,
) -> None:
    """Clean and standardise the already-wide Market Depth export.

    WHY Parquet output via incremental ParquetWriter (not CSV append):
    this feed has ~40 depth-level columns (Lx-BidPrice/BidSize/AskPrice/
    AskSize), many of which are sparsely populated -- deeper levels
    (L5-L10) are frequently empty, especially for a less liquid stock.
    Writing that as repeated literal CSV text ("", "NaN", ...) wastes an
    enormous amount of disk space compared to Parquet's columnar,
    compressed storage, which was confirmed in practice: the CSV-append
    version of this function ran the machine out of disk space partway
    through the file. `pq.ParquetWriter` lets us write one chunk (one
    "row group") at a time, keeping the same bounded-memory streaming
    behaviour as before, while producing a much smaller file on disk.
    """
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    n_chunks = 0

    try:
        for chunk in pd.read_csv(
            input_path, chunksize=chunksize, dtype=str, keep_default_na=False, low_memory=False
        ):
            n_chunks += 1

            chunk["ric"] = chunk[RIC_COL].apply(normalize_ric)
            ts_utc, ts_local = parse_timestamps_vectorized(chunk)
            chunk["ts_utc"] = ts_utc
            chunk["ts_local"] = ts_local
            chunk["type"] = chunk[TYPE_COL]
            chunk["event_id"] = make_event_id_vectorized(chunk)

            level_cols = [
                c for c in chunk.columns
                if c.startswith("L") and ("Price" in c or "Size" in c)
            ]
            for col in level_cols:
                chunk[col] = coerce_numeric(chunk[col])

            keep_cols = ["event_id", "ric", "ts_utc", "ts_local", "type"] + level_cols
            out_df = chunk[keep_cols]

            table = pa.Table.from_pandas(out_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)

            log.info("Processed chunk %d (%d rows), %d depth-level columns detected.",
                      n_chunks, len(chunk), len(level_cols))
    finally:
        if writer is not None:
            writer.close()

    log.info("Done. Prepared Market Depth output written to %s", output_path)


# ----------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the whole data preparation from the submission folder with
    #     python scripts/rebuild_data.py
    # which calls this module with the correct input and output paths.
    raise SystemExit("Run `python scripts/rebuild_data.py` from the submission folder (see README.md).")
