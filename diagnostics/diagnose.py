"""
diagnose.py -- standalone sanity check for the TX/MD loaders.

Run directly from Command Prompt (no R/Quarto/reticulate needed):

    cd "C:\\Users\\Lungelo Phewa\\Documents\\Data Science\\Data Analysis for Trading\\Assignments"
    python diagnose.py

Edit the three paths below to match your setup, then run.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # so `import data_io` finds the module
import data_io

TX_TRIMMED_PATH = Path(r"C:\Users\Lungelo Phewa\Documents\Data Science\Data Analysis for Trading\Assignments\AGL_GRT_TX_trimmed.csv")
TX_FLAT_PATH = Path(r"C:\Users\Lungelo Phewa\Documents\Data Science\Data Analysis for Trading\Assignments\AGL_GRT_TX_flat.csv")
MD_PATH = Path(r"C:\Users\Lungelo Phewa\Documents\Data Science\Data Analysis for Trading\Assignments\AGL_GRT_MD_flat.parquet")
SIGN_PATH = Path(r"C:\Users\Lungelo Phewa\Documents\Data Science\Data Analysis for Trading\Assignments\AGL_GRT_stage4_final.parquet")
RICS = ["AGLJ.J", "GRTJ.J"]


def daily_coverage(df, ric, label):
    """Per-day min/max/count of local SAST trade time for one ric, plus a
    verdict on whether the last-trade time is essentially fixed across days
    (-> systematic export cutoff) or varies a lot (-> day-specific cause)."""
    print(f"\n--- Daily coverage: {label} ({ric}) ---")
    sub = df[df["ric"] == ric].copy()
    if sub.empty:
        print("  no rows for this ric in this file")
        return None
    sub["local_time"] = sub["ts"].dt.tz_convert(data_io.TZ_LOCAL).dt.time
    span = sub.groupby("date")["local_time"].agg(["min", "max", "count"])
    print(span.to_string())

    # Convert datetime.time -> seconds-since-midnight directly, rather than
    # round-tripping through a string + strptime format (which breaks the
    # moment microseconds are present, as they are in the real data).
    def _secs(t):
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

    last_secs = span["max"].apply(_secs)
    std = pd.Timedelta(seconds=last_secs.std())
    latest_time = span.loc[last_secs.idxmax(), "max"]
    earliest_time = span.loc[last_secs.idxmin(), "max"]
    print(f"\n  Last-trade-of-day time: min={earliest_time}, max={latest_time}, "
          f"std across days={std}")
    verdict = "FIXED cutoff" if std < pd.Timedelta(minutes=10) else "varies day to day"
    print(f"  Verdict: {verdict}")
    return latest_time, std

print("Peeking SIGN_PATH schema (cheap, no data load)...")
try:
    cols = data_io._peek_parquet_columns(SIGN_PATH)
    print(f"  {len(cols)} columns in {SIGN_PATH.name}: {cols}")
except Exception as e:
    print(f"  Could not peek schema: {e}")

print("\nLoading TX (trimmed)...")
trades_trimmed = data_io.load_tx(TX_TRIMMED_PATH, RICS, external_sign_path=SIGN_PATH)
print("Loading TX (flat, untrimmed)...")
trades_flat = data_io.load_tx(TX_FLAT_PATH, RICS)  # no sign file needed for a coverage check
print("Loading MD...")
md = data_io.load_md(MD_PATH, RICS)

print("\n--- MD diagnostics ---")
print("ric values:", repr(md["ric"].unique()))
md_dates = sorted(md["date"].unique())
print(f"date range: {md_dates[0]} to {md_dates[-1]}  ({len(md_dates)} distinct days)")

print("\n" + "=" * 70)
print("COVERAGE COMPARISON: trimmed TX vs flat TX vs MD")
print("=" * 70)
trimmed_last, trimmed_std = daily_coverage(trades_trimmed, "AGLJ.J", "TX_trimmed")
flat_last, flat_std = daily_coverage(trades_flat, "AGLJ.J", "TX_flat")
daily_coverage(md, "AGLJ.J", "MD")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if flat_last > trimmed_last:
    print(f"TX_flat's last trade ({flat_last}) is LATER than TX_trimmed's ({trimmed_last}).")
    print("-> AGL_GRT_TX_flat.csv is the better source. Point TX_PATH at it in the")
    print("   .qmd (you've already done this) AND make sure to fully re-run every")
    print("   downstream chunk -- or just Restart R and re-render top to bottom --")
    print("   since changing a path variable alone does not reload data already")
    print("   sitting in an active Python session.")
else:
    print(f"TX_flat's last trade ({flat_last}) is NOT later than TX_trimmed's ({trimmed_last}).")
    print("-> Both files share the same cutoff -- this points to a genuine gap in")
    print("   the original Tick History pull itself, not Assignment 1's trimming")
    print("   step. Worth checking MD's own cutoff above: if MD covers the full")
    print("   session but TX doesn't on either file, that's an asymmetry between")
    print("   the two source feeds to document as a known data limitation.")

print("\n--- Overlap check (trimmed TX vs MD; days present in BOTH feeds) ---")
tx_dates = sorted(trades_trimmed["date"].unique())
overlap = sorted(set(tx_dates) & set(md_dates))
print(f"{len(overlap)} overlapping days")
if overlap:
    example_day = overlap[0]
    n_trades = trades_trimmed[(trades_trimmed["ric"] == "AGLJ.J") &
                               (trades_trimmed["date"] == example_day)].shape[0]
    print(f"Suggested EXAMPLE_DAY = {example_day!r}  (AGLJ.J has {n_trades} trades on this day)")
