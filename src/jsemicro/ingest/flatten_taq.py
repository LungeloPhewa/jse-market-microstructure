"""
Refinitiv TAQ Flattening Pipeline -- Transactions feed (vectorized)
======================================================================

Converts the raw Transactions export (header row + 'FID' attribute rows,
confirmed against real sample data) into a clean, wide, time-ordered event
table -- one row per event, ready for microstructure feature engineering.

PERFORMANCE NOTE
------------------
This version replaces the original per-row `.apply(axis=1)` / `iterrows()`
based flattening with vectorized pandas operations (boolean masks,
`pivot_table`, vectorized string/datetime ops). The original approach
measured at roughly 4 minutes per 200,000-row chunk -- projecting to
~5+ hours for the full 15.79M-row file. Row-at-a-time Python loops are
inherently slow because each row triggers Python-level function-call
overhead; pandas' vectorized operations instead push the work down into
compiled C code operating on whole columns at once, which is typically
one to two orders of magnitude faster for this kind of reshaping.

CONFIRMED SCHEMA (from real sample data)
------------------------------------------
Columns: #RIC, Date[G], Time[G], GMT Offset, Type, MSG/FID Number,
         Message Type, FID Name, FID Value, FID Enum String, PE Code,
         Sequence Number, Source RIC, Number of FIDs, ...

A HEADER row (starts a new event) has Type != "FID" (e.g. "Raw-TAS"),
non-null #RIC / Date[G] / Time[G] / GMT Offset, and a Message Type
(e.g. "UPDATE_PC"). Time[G] is NOT consistently formatted -- some rows
include fractional seconds (e.g. "02:30:00.917"), others don't
(e.g. "07:09:43") -- timestamp parsing below tries both.

A CONTINUATION (FID) row has Type == "FID", #RIC/Date[G]/Time[G]/GMT
Offset all NA, and carries one attribute per row via the (FID Name,
FID Value) pair, e.g. ("TRDPRC_1", 38019.7), ("SEQNUM", 7736).

Market Depth is handled separately (see market_depth_prepare.py) because
that feed arrives ALREADY wide -- it does not need this FID-pivoting step.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
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
MSG_TYPE_COL = "Message Type"
FID_NAME_COL = "FID Name"
FID_VALUE_COL = "FID Value"

FID_ROW_TYPE = "FID"

NUMERIC_FIELDS = ["TRDPRC_1", "NETCHNG_1", "PRCTCK_1", "PCTCHNG", "NUM_MOVES", "GEN_VAL2", "SEQNUM"]

JSE_TZ = ZoneInfo("Africa/Johannesburg")
JSE_FIXED_OFFSET_HOURS = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Step 1: Stream the raw file in chunks
# ----------------------------------------------------------------------------

def stream_chunks(path: str | Path, chunksize: int = CHUNK_SIZE, compression: str = "infer"):
    """Yield raw chunks of the file rather than loading it all at once."""
    for chunk in pd.read_csv(
        path, chunksize=chunksize, dtype=str, keep_default_na=False,
        low_memory=False, compression=compression,
    ):
        yield chunk


# ----------------------------------------------------------------------------
# Step 2: Vectorized header detection and chunk-boundary stitching
# ----------------------------------------------------------------------------

def header_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series: True where a row starts a new event.

    WHY vectorized: a single column comparison (`!=`) runs as one
    compiled operation over the whole column, versus calling a Python
    function once per row via `.apply(axis=1)` -- this alone accounts
    for a large share of the original slowness, since it runs on EVERY
    row in the file, header or not.
    """
    return df[TYPE_COL] != FID_ROW_TYPE


def split_trailing_incomplete_event(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a chunk into (safely-complete events, possibly-incomplete tail)."""
    is_header = header_mask(df)
    header_positions = df.index[is_header]

    if len(header_positions) == 0:
        return df.iloc[0:0], df

    last_header_idx = header_positions[-1]
    complete = df.loc[: last_header_idx - 1] if last_header_idx > df.index[0] else df.iloc[0:0]
    carry = df.loc[last_header_idx:]
    return complete, carry


def stitch_boundary(carry_over: pd.DataFrame, chunk: pd.DataFrame) -> pd.DataFrame:
    """Prepend last chunk's carried-over (possibly incomplete) event rows."""
    if carry_over is not None and not carry_over.empty:
        return pd.concat([carry_over, chunk], ignore_index=True)
    return chunk


# ----------------------------------------------------------------------------
# Step 3: Vectorized flatten -- the main performance fix
# ----------------------------------------------------------------------------

def normalize_ric_series(s: pd.Series) -> pd.Series:
    return s.str.strip().str.upper()


def parse_timestamps_vectorized(date_series: pd.Series, time_series: pd.Series, offset_series: pd.Series):
    """Vectorized UTC/local timestamp construction, tolerant of both
    Time[G] formats seen in the real data (with/without fractional
    seconds). Same logic as before, but operating on whole Series at
    once instead of one (date_str, time_str) pair per Python call.
    """
    combined = date_series + " " + time_series

    naive = pd.to_datetime(combined, format="%d-%b-%Y %H:%M:%S.%f", errors="coerce")
    missing = naive.isna()
    if missing.any():
        naive.loc[missing] = pd.to_datetime(combined.loc[missing], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    missing = naive.isna()
    if missing.any():
        naive.loc[missing] = pd.to_datetime(combined.loc[missing], errors="coerce")

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
    _stated = pd.to_numeric(offset_series, errors="coerce").dropna()
    if len(_stated) and not (_stated == JSE_FIXED_OFFSET_HOURS).all():
        raise ValueError(f"Unexpected GMT offsets in the file: {sorted(_stated.unique())}")
    return ts_utc, ts_local


def coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def discover_fid_names(input_path: str | Path, chunksize: int = CHUNK_SIZE, compression: str = "infer") -> list[str]:
    """First pass: collect every distinct FID Name that appears anywhere.

    WHY THIS PASS IS NECESSARY: pivoting per-chunk only creates columns
    for the FID Names encountered in THAT chunk -- different chunks
    contain different mixes of event types and therefore produce
    DIFFERENT column sets. Appending these mismatched per-chunk
    DataFrames directly to one CSV silently produces a file with a
    different number of fields per row depending on which chunk a row
    came from -- exactly what the field-count diagnostic surfaced (18
    distinct field counts, from 16 to 376, in a file that should have
    exactly one). Knowing the full, fixed universe of FID Names up front
    lets every chunk be reindexed onto the SAME columns before writing,
    which is what makes incremental append-mode CSV writing safe.

    This does mean reading the file twice overall (once here, once in
    the main flatten pass) -- but since this pass only needs the
    "FID Name" column (via `usecols`), it's much cheaper than a full
    flatten pass, and the two-pass cost is trivial next to the
    5-hour-to-2-minute speedup already achieved.
    """
    fid_names: set[str] = set()
    for chunk in pd.read_csv(
        input_path, chunksize=chunksize, dtype=str, keep_default_na=False,
        low_memory=False, compression=compression, usecols=[FID_NAME_COL],
    ):
        vals = chunk[FID_NAME_COL]
        vals = vals[~vals.isin(("", "NA"))]
        fid_names.update(vals.unique().tolist())
    fid_columns = sorted(fid_names)
    log.info("Discovered %d distinct FID Names across the file.", len(fid_columns))
    return fid_columns


def flatten_chunk_vectorized(df: pd.DataFrame, seq_counter: dict, fid_columns: list[str]) -> pd.DataFrame:
    """Flatten one block of complete-event rows into one row per event,
    with a FIXED, IDENTICAL set of output columns on every call.

    WHY the fixed `fid_columns` parameter: this is the actual fix for the
    ragged-CSV bug -- every chunk is reindexed onto the exact same column
    list (in the exact same order), regardless of which FID Names that
    particular chunk's events happened to use, so appending chunks to one
    CSV file remains schema-safe throughout.
    """
    is_header = header_mask(df)
    group_ids = is_header.cumsum()
    df = df.assign(_group_id=group_ids)

    headers = df.loc[is_header].set_index("_group_id")
    fid_rows = df.loc[~is_header]
    fid_rows = fid_rows.loc[~fid_rows[FID_NAME_COL].isin(("", "NA"))]

    if not fid_rows.empty:
        fid_wide = fid_rows.pivot_table(
            index="_group_id", columns=FID_NAME_COL, values=FID_VALUE_COL, aggfunc="first"
        )
        wide = headers.join(fid_wide, how="left")
    else:
        wide = headers.copy()

    # Guarantee every known FID column is present, even if this chunk had
    # no events using it -- this, plus the fixed final column order below,
    # is what keeps every chunk's output schema identical.
    for col in fid_columns:
        if col not in wide.columns:
            wide[col] = pd.NA

    wide["ric"] = normalize_ric_series(wide[RIC_COL])
    ts_utc, ts_local = parse_timestamps_vectorized(wide[DATE_COL], wide[TIME_COL], wide[OFFSET_COL])
    wide["ts_utc"] = ts_utc
    wide["ts_local"] = ts_local
    wide["type"] = wide[TYPE_COL]
    wide["message_type"] = wide[MSG_TYPE_COL] if MSG_TYPE_COL in wide.columns else ""

    key = wide["ric"].astype(str) + "|" + wide[DATE_COL] + "|" + wide[TIME_COL]
    seqs = []
    for k in key:
        seq_counter[k] = seq_counter.get(k, -1) + 1
        seqs.append(seq_counter[k])
    wide["event_id"] = [
        hashlib.sha1(f"{k}|{s}".encode("utf-8")).hexdigest()[:16] for k, s in zip(key, seqs)
    ]

    # Fixed final column order -- identity columns first, then every FID
    # column in the same (alphabetical) order every single time.
    identity_cols = ["event_id", "ric", "ts_utc", "ts_local", "type", "message_type"]
    wide = wide.reindex(columns=identity_cols + fid_columns)

    return wide.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Step 4: Incremental flush to disk
# ----------------------------------------------------------------------------

def _flush(wide: pd.DataFrame, output_path: str | Path, first_write: bool) -> None:
    for col in NUMERIC_FIELDS:
        if col in wide.columns:
            wide[col] = coerce_numeric(wide[col])
    wide.to_csv(output_path, mode="a", header=first_write, index=False)
    log.info("Flushed %d events -> %s", len(wide), output_path)


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def flatten_taq_file(
    input_path: str | Path,
    output_path: str | Path,
    chunksize: int = CHUNK_SIZE,
) -> None:
    """Flatten a raw Transactions export into a wide, one-row-per-event CSV.

    Runs TWO passes over the file: first `discover_fid_names` to build
    the fixed column schema, then the actual flatten -- see
    `discover_fid_names` and `flatten_chunk_vectorized` docstrings for why
    this two-pass approach is required for a schema-consistent output.
    """
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    log.info("Pass 1/2: discovering the full set of FID Names...")
    fid_columns = discover_fid_names(input_path, chunksize)

    log.info("Pass 2/2: flattening events onto the fixed schema...")
    carry_over = pd.DataFrame()
    seq_counter: dict = {}
    first_write = True
    n_chunks = 0

    for raw_chunk in stream_chunks(input_path, chunksize):
        n_chunks += 1
        chunk = stitch_boundary(carry_over, raw_chunk)
        complete, carry_over = split_trailing_incomplete_event(chunk)

        if not complete.empty:
            wide = flatten_chunk_vectorized(complete, seq_counter, fid_columns)
            if not wide.empty:
                _flush(wide, output_path, first_write)
                first_write = False

        log.info("Processed chunk %d (%d raw rows)", n_chunks, len(raw_chunk))

    if not carry_over.empty:
        wide = flatten_chunk_vectorized(carry_over, seq_counter, fid_columns)
        if not wide.empty:
            _flush(wide, output_path, first_write)

    log.info("Done. Flattened output written to %s", output_path)


# ----------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the whole data preparation from the submission folder with
    #     python scripts/rebuild_data.py
    # which calls this module with the correct input and output paths.
    raise SystemExit("Run `python scripts/rebuild_data.py` from the submission folder (see README.md).")
