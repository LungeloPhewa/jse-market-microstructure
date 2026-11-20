"""
inspect_nat_rows.py
--------------------
Pulls out and characterizes every row with a missing (NaT) ts_utc in the
merged dataset, to understand what these 10,151 rows actually are before
deciding how to handle them in Stage 4 cleaning.

Reads the merged file row-group by row-group (memory-bounded) rather than
loading all 9.4M rows at once, since we only need the ~10k matching rows.

Usage from Command Prompt:

    python inspect_nat_rows.py
"""

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# --------------------------------------------------------------------
from ..config import PATHS

MERGED_PATH = PATHS.merged_sorted
OUTPUT_CSV = PATHS.out_dir / "nat_rows_inspection.csv"
# --------------------------------------------------------------------


def main():
    pf = pq.ParquetFile(str(MERGED_PATH))
    print(f"Scanning {pf.num_row_groups} row groups for NaT ts_utc rows...")

    matches = []
    total_scanned = 0
    for i, batch in enumerate(pf.iter_batches(batch_size=100_000)):
        df = batch.to_pandas()
        total_scanned += len(df)
        hit = df[df["ts_utc"].isna()]
        if len(hit):
            matches.append(hit)
        if (i + 1) % 20 == 0:
            print(f"  ...scanned {total_scanned:,} rows, {sum(len(m) for m in matches):,} matches so far")

    if not matches:
        print("No NaT rows found -- unexpected given the earlier count of 10,151. "
              "Check MERGED_PATH points at the right file.")
        return

    nat_df = pd.concat(matches, ignore_index=True)
    print()
    print(f"Total NaT ts_utc rows found: {len(nat_df):,} (expected ~10,151)")
    print()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # Characterize by whatever categorical columns exist
    for col in ["ric", "type", "source", "message_type"]:
        if col in nat_df.columns:
            print(f"--- Breakdown by '{col}' ---")
            print(nat_df[col].value_counts(dropna=False).to_string())
            print()

    # Check ts_local -- do these rows have a valid LOCAL timestamp even
    # though ts_utc is missing? If so, this might just be a timezone
    # conversion artifact rather than a genuinely missing timestamp.
    if "ts_local" in nat_df.columns:
        n_local_valid = nat_df["ts_local"].notna().sum()
        print(f"Of these {len(nat_df):,} NaT-ts_utc rows, {n_local_valid:,} have a VALID ts_local.")
        if n_local_valid > 0:
            print("-> This suggests a UTC conversion issue (e.g. an unparseable "
                  "local timestamp/offset) rather than a truly missing timestamp. "
                  "These might be recoverable by re-deriving ts_utc from ts_local "
                  "instead of dropping the rows outright.")
        print()

    # Check SEQNUM presence -- are these rows quotes, trades, or something else?
    if "SEQNUM" in nat_df.columns:
        n_seqnum = nat_df["SEQNUM"].notna().sum()
        print(f"Of these rows, {n_seqnum:,} have a non-null SEQNUM "
              f"(SEQNUM usually only appears on trade/TX rows, not quotes).")
        print()

    # Show a sample of full rows for manual inspection
    cols_to_show = [c for c in ["ric", "ts_utc", "ts_local", "SEQNUM", "event_id",
                                 "source", "type", "message_type",
                                 "BID", "ASK", "BIDSIZE", "ASKSIZE",
                                 "TRDPRC_1", "TRDVOL_1"] if c in nat_df.columns]
    print("--- Sample of 20 NaT rows (full columns) ---")
    print(nat_df[cols_to_show].head(20).to_string())
    print()

    nat_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Full set of {len(nat_df):,} NaT rows written to: {OUTPUT_CSV}")
    print("(Open this in Excel if you want to eyeball more of them directly.)")


if __name__ == "__main__":
    main()
