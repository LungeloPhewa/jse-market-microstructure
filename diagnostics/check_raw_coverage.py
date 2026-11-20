"""
check_raw_coverage.py -- does the RAW Refinitiv TAQ export itself contain
data past ~15:50 SAST, or does the limitation already exist before
flatten_taq.py ever touches it?

This reads ONLY the four columns needed (#RIC, Date[G], Time[G], Type) in
chunks, so it's cheap even against a 15M+ row file, and completely
bypasses flatten_taq.py and data_io.py -- if a late cutoff shows up here
too, the limitation is in the raw Tick History pull itself, upstream of
everything built so far.

Run:
    python check_raw_coverage.py
Edit RAW_PATH and TARGET_DATE below first.
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(r"C:\Users\Lungelo Phewa\Documents\Data Science\Data Analysis for Trading\Assignments\JSE-AGL-GRT-Transactions-Raw-1SEP2013-1OCT2013-N128101748.csv.gz")
TARGET_RIC = "AGLJ.J"
# Check a few different days -- pick ones that showed different trade
# counts in the earlier daily_coverage table, to make sure this isn't
# specific to one unusual day.
TARGET_DATES = ["02-SEP-2013", "10-SEP-2013", "20-SEP-2013"]  # match Date[G]'s raw format
CHUNK_SIZE = 500_000

RIC_COL = "#RIC"
DATE_COL = "Date[G]"
TIME_COL = "Time[G]"
TYPE_COL = "Type"

print(f"Scanning {RAW_PATH.name} in {CHUNK_SIZE:,}-row chunks "
      f"(reading only {RIC_COL}, {DATE_COL}, {TIME_COL}, {TYPE_COL})...")
print(f"Looking for ric={TARGET_RIC}, dates={TARGET_DATES}\n")

# This file is gzip-compressed on disk despite its visible name ending in
# .csv (confirmed via Explorer's Properties dialog: "Type of file: ...
# (.gz)"). compression="infer" would only detect this from a literal
# ".gz" suffix in the path, which isn't present here, so gzip is forced
# explicitly rather than relying on name-based inference.
found = {d: {"max_time": None, "n_rows": 0, "last_type": None} for d in TARGET_DATES}
total_rows_scanned = 0

for i, chunk in enumerate(pd.read_csv(
    RAW_PATH, chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False,
    low_memory=False, usecols=[RIC_COL, DATE_COL, TIME_COL, TYPE_COL],
    compression="gzip",
)):
    total_rows_scanned += len(chunk)
    # Header rows only -- #RIC is populated only on header (non-FID) rows,
    # per flatten_taq.py's own documented schema.
    sub = chunk[(chunk[RIC_COL].str.strip().str.upper() == TARGET_RIC) &
                (chunk[DATE_COL].isin(TARGET_DATES))]
    for d in TARGET_DATES:
        day_rows = sub[sub[DATE_COL] == d]
        if day_rows.empty:
            continue
        found[d]["n_rows"] += len(day_rows)
        day_max_time = day_rows[TIME_COL].max()  # lexicographic max works for HH:MM:SS[.ffffff]
        if found[d]["max_time"] is None or day_max_time > found[d]["max_time"]:
            found[d]["max_time"] = day_max_time
            found[d]["last_type"] = day_rows.loc[day_rows[TIME_COL] == day_max_time, TYPE_COL].iloc[0]
    if (i + 1) % 10 == 0:
        print(f"  ...scanned {total_rows_scanned:,} raw rows so far")

print(f"\nTotal raw rows scanned: {total_rows_scanned:,}\n")
print("--- Results (RAW file, before any flattening) ---")
for d in TARGET_DATES:
    r = found[d]
    if r["max_time"] is None:
        print(f"{d}: NO rows found for {TARGET_RIC} -- check the date format matches Date[G] exactly")
    else:
        print(f"{d}: {r['n_rows']:,} header rows, latest Time[G] = {r['max_time']}  "
              f"(Type at that time: {r['last_type']!r})")

print("\nIf these latest times are still ~13:2X-13:5X (i.e. ~15:2X-15:5X SAST, "
      "accounting for the +2h JSE offset), the raw export itself stops there -- "
      "this is a genuine upstream data limitation, not a bug in flatten_taq.py "
      "or anything built downstream of it. If instead the raw file has rows "
      "much later than that (e.g. up to ~14:30 UTC / 16:30 SAST, matching MD's "
      "own cutoff), then flatten_taq.py IS dropping them somewhere, and that's "
      "worth investigating further.")
