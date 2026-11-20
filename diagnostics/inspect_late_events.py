"""
inspect_late_events.py -- is the ~15:50 SAST cutoff in AGL_GRT_TX_flat.csv
a real feature of the raw data, or something flatten_taq.py is dropping?

This reuses flatten_taq.py's OWN functions (stream_chunks, stitch_boundary,
split_trailing_incomplete_event, flatten_chunk_vectorized) to run the exact
same flattening logic against the real raw file -- but instead of writing
every event to disk, it keeps only rows matching one target (ric, date) in
memory, so we can inspect them directly without re-implementing (and
possibly re-introducing different bugs into) the flattening logic ourselves.

For every event on/after CUTOFF_TIME, it reports whether a genuine
TRDPRC_1 trade price is attached, or whether it's some other event type
(quote, correction, etc.) that data_io.py's own trade filter
(TRDPRC_1 notna & > 0) would correctly exclude downstream.

Run:
    python inspect_late_events.py
Requires flatten_taq.py in the same folder (or on PYTHONPATH).
Edit RAW_PATH / TARGET_RIC / TARGET_DATE / CUTOFF_TIME below first.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import flatten_taq as ft

RAW_PATH = Path(r"C:\Users\Lungelo Phewa\Documents\Data Science\Data Analysis for Trading\Assignments\JSE-AGL-GRT-Transactions-Raw-1SEP2013-1OCT2013-N128101748.csv.gz")
TARGET_RIC = "AGLJ.J"
TARGET_DATE = "02-SEP-2013"   # match Date[G]'s raw format exactly
CUTOFF_TIME = "15:00:00"      # local SAST; events at/after this are "late session"
CHUNK_SIZE = 200_000


def main():
    # flatten_chunk_vectorized's output keeps ts_utc/ts_local (proper
    # timestamps) but drops the raw Date[G]/Time[G] string columns -- so
    # filter/display using ts_local throughout, not the raw column names.
    target_date_obj = pd.to_datetime(TARGET_DATE, format="%d-%b-%Y").date()
    cutoff_time_obj = pd.to_datetime(CUTOFF_TIME, format="%H:%M:%S").time()

    print("Pass 1/2: discovering FID columns (same as flatten_taq.py)...")
    fid_columns = ft.discover_fid_names(RAW_PATH, CHUNK_SIZE)

    print(f"\nPass 2/2: running the REAL flatten pipeline, keeping only "
          f"{TARGET_RIC} / {TARGET_DATE} events from each chunk's output "
          f"(discarding everything else to keep memory low)...")

    carry_over = pd.DataFrame()
    seq_counter: dict = {}
    kept_rows = []
    n_chunks = 0

    def _filter(wide):
        return wide[(wide["ric"] == TARGET_RIC) & (wide["ts_local"].dt.date == target_date_obj)]

    for raw_chunk in ft.stream_chunks(RAW_PATH, CHUNK_SIZE):
        n_chunks += 1
        chunk = ft.stitch_boundary(carry_over, raw_chunk)
        complete, carry_over = ft.split_trailing_incomplete_event(chunk)
        if not complete.empty:
            wide = ft.flatten_chunk_vectorized(complete, seq_counter, fid_columns)
            if not wide.empty:
                match = _filter(wide)
                if not match.empty:
                    kept_rows.append(match)
        if n_chunks % 20 == 0:
            print(f"  ...processed {n_chunks} chunks, {sum(len(k) for k in kept_rows)} "
                  f"matching events kept so far")

    if not carry_over.empty:
        wide = ft.flatten_chunk_vectorized(carry_over, seq_counter, fid_columns)
        if not wide.empty:
            match = _filter(wide)
            if not match.empty:
                kept_rows.append(match)

    if not kept_rows:
        print(f"\nNo events found at all for {TARGET_RIC} / {TARGET_DATE} -- "
              f"double check TARGET_DATE matches Date[G]'s exact raw format.")
        return

    day = pd.concat(kept_rows, ignore_index=True).sort_values("ts_local")
    day["local_time"] = day["ts_local"].dt.time

    print(f"\n{'='*70}\nRESULTS: {TARGET_RIC} on {TARGET_DATE}\n{'='*70}")
    print(f"Total flattened events (ALL types, via flatten_taq.py's real logic): {len(day)}")
    print(f"Local time range: {day['local_time'].min()} to {day['local_time'].max()}")

    late = day[day["local_time"] >= cutoff_time_obj]
    print(f"\nEvents at/after {CUTOFF_TIME} local: {len(late)}")

    if "TRDPRC_1" in late.columns:
        price_numeric = pd.to_numeric(late["TRDPRC_1"], errors="coerce")
        has_price = price_numeric.notna() & (price_numeric > 0)
        print(f"  -> have a genuine TRDPRC_1 trade price:     {has_price.sum()}")
        print(f"  -> do NOT (quote/correction/other types):   {(~has_price).sum()}")
    else:
        has_price = pd.Series([False] * len(late), index=late.index)
        print("  TRDPRC_1 column not present at all among late events")

    if "message_type" in late.columns:
        print(f"\nMessage Type breakdown for late events:")
        print(late["message_type"].value_counts().to_string())

    if has_price.any():
        latest_real_trade = late.loc[has_price, "local_time"].max()
        print(f"\n>> Latest event with a GENUINE trade price: {latest_real_trade}")
        print(">> If this is still ~15:5X, real trades themselves stop there --")
        print(">> flatten_taq.py and data_io.py are both behaving correctly, and")
        print(">> this is simply a real feature of the market/data (e.g. the last")
        print(">> executed trade of the day happened to print around then).")
    else:
        print(f"\n>> NO late event has a genuine trade price at all -- consistent")
        print(f">> with real trading activity itself stopping before {CUTOFF_TIME},")
        print(f">> while non-trade events (quotes etc.) continue later under the")
        print(f">> same raw 'Type' label.")

    print(f"\nSample of late events (first 15, sorted by time):")
    cols_to_show = ["ric", "local_time", "message_type"]
    for c in ("TRDPRC_1", "TRDVOL_1"):
        if c in late.columns:
            cols_to_show.append(c)
    print(late[cols_to_show].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
