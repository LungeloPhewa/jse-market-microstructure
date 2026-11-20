"""
diagnose_merge_violation.py
-----------------------------
Inspects the ACTUAL DATA around a known ordering violation, using files
already on disk (the completed merged output + the original 94 Stage 2
chunks). Does NOT re-run the 35-minute merge.

Two things this does:

1. Prints all columns (not just the sort key) for a window of rows
   around the violation point in the already-merged output file, using
   row-group-aware access so it doesn't need to scan the whole 9.4M-row
   file.

2. Scans all 94 original chunk files for any rows matching the same
   `ric` within a time window around the violation, to see exactly how
   many different chunks contributed data in that window and what their
   relative order looks like BEFORE the merge combined them.

Usage from Command Prompt:

    python diagnose_merge_violation.py
"""

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# --------------------------------------------------------------------
# EDIT IF NEEDED -- these match the violation you already found
# --------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\Lungelo Phewa\Documents\Data Science\Data Analysis for Trading\Assignment1")
CHUNKS_DIR = BASE_DIR / "taq_stage2_chunks"
MERGED_PATH = BASE_DIR / "AGL_GRT_merged_sorted.parquet"

VIOLATION_ROW = 863915          # 0-indexed row number reported by verify_sorted
RIC = "AGLJ.J"
TS_WINDOW_LOW = pd.Timestamp("2013-09-04 07:46:00", tz="UTC")
TS_WINDOW_HIGH = pd.Timestamp("2013-09-04 07:47:00", tz="UTC")

WRITE_BATCH_SIZE = 50_000       # must match taq_stage3_merge.WRITE_BATCH_SIZE used for the run
CONTEXT_ROWS = 15               # rows of context to show before/after the violation
# --------------------------------------------------------------------


def inspect_merged_output_window():
    print("=" * 78)
    print(f"PART 1: rows around index {VIOLATION_ROW} in the merged output file")
    print("=" * 78)

    row_group_idx = VIOLATION_ROW // WRITE_BATCH_SIZE
    local_offset = VIOLATION_ROW % WRITE_BATCH_SIZE

    pf = pq.ParquetFile(str(MERGED_PATH))
    print(f"Merged file has {pf.num_row_groups} row groups, ~{WRITE_BATCH_SIZE} rows each.")
    print(f"Violation row {VIOLATION_ROW} -> row group {row_group_idx}, local offset {local_offset}")

    # Read the row group containing the violation. If it's very close to a
    # row-group boundary, also grab the previous/next row group for context.
    groups_to_read = [row_group_idx]
    if local_offset < CONTEXT_ROWS and row_group_idx > 0:
        groups_to_read.insert(0, row_group_idx - 1)
    if local_offset > WRITE_BATCH_SIZE - CONTEXT_ROWS and row_group_idx < pf.num_row_groups - 1:
        groups_to_read.append(row_group_idx + 1)

    table = pf.read_row_groups(groups_to_read)
    df = table.to_pandas()

    # Figure out where VIOLATION_ROW falls within this concatenated slice
    start_row_of_slice = groups_to_read[0] * WRITE_BATCH_SIZE
    local_idx_in_slice = VIOLATION_ROW - start_row_of_slice

    lo = max(0, local_idx_in_slice - CONTEXT_ROWS)
    hi = min(len(df), local_idx_in_slice + CONTEXT_ROWS)

    cols_to_show = [c for c in ["ric", "ts_utc", "SEQNUM", "event_id", "source", "type",
                                 "message_type", "BID", "ASK", "TRDPRC_1", "TRDVOL_1"]
                     if c in df.columns]

    window = df.iloc[lo:hi][cols_to_show].reset_index(drop=True)
    window.index = range(start_row_of_slice + lo, start_row_of_slice + hi)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print()
    print(f"Showing global rows {start_row_of_slice + lo} to {start_row_of_slice + hi - 1} "
          f"(violation is at row {VIOLATION_ROW}):")
    print(window.to_string())
    print()


def inspect_source_chunks():
    print("=" * 78)
    print(f"PART 2: which of the 94 original chunks contain '{RIC}' rows in "
          f"[{TS_WINDOW_LOW}, {TS_WINDOW_HIGH}]")
    print("=" * 78)
    print()

    chunks = sorted(CHUNKS_DIR.glob("sorted_chunk_*.parquet"))
    all_hits = []

    cols_needed = ["ric", "ts_utc", "SEQNUM", "event_id"]
    for c in chunks:
        try:
            df = pd.read_parquet(c, columns=cols_needed)
        except Exception as e:
            print(f"  (skipping {c.name}: could not read columns -- {e})")
            continue
        mask = (df["ric"] == RIC) & (df["ts_utc"] >= TS_WINDOW_LOW) & (df["ts_utc"] <= TS_WINDOW_HIGH)
        sub = df[mask].copy()
        if len(sub):
            sub["__chunk_file"] = c.name
            all_hits.append(sub)

    if not all_hits:
        print(f"No rows found for {RIC} in this time window across any chunk. "
              f"Double-check RIC / time window values at the top of this script.")
        return

    combined = pd.concat(all_hits, ignore_index=True)
    combined = combined.sort_values(["ts_utc", "SEQNUM"], kind="mergesort", na_position="last")

    n_chunks_involved = combined["__chunk_file"].nunique()
    print(f"Found {len(combined)} rows across {n_chunks_involved} different chunk file(s) "
          f"in this time window:")
    print(f"  Chunks involved: {sorted(combined['__chunk_file'].unique())}")
    print()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(combined.to_string(index=False))
    print()

    if n_chunks_involved > 1:
        print("-> Multiple chunks overlap in this time window. That's NORMAL for an "
              "external merge sort (each chunk covers a different, possibly overlapping, "
              "slice of the original read order) -- the k-way merge is specifically "
              "responsible for correctly interleaving them by timestamp. If the ABOVE "
              "list (sorted correctly, straight from the source chunks) looks right, "
              "but PART 1's merged output doesn't match this order, that confirms a "
              "bug in the merge's cross-chunk interleaving logic itself.")
    else:
        print("-> Only ONE chunk has rows in this window, which is unexpected if the "
              "merge is genuinely interleaving multiple sources here. Worth double "
              "checking the time window / RIC above are exactly right.")


def main():
    inspect_merged_output_window()
    inspect_source_chunks()


if __name__ == "__main__":
    main()
