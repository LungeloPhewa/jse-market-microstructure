"""
run_option_b.py
-----------------
Re-runs the merge using AGL_GRT_MD_flat.parquet (the un-filtered Market
Depth file, confirmed to extend to ~16:30 local) instead of the missing
AGL_GRT_MD_filtered.parquet, recovering the ~90 minutes/day of quote
coverage lost to the untraceable earlier filtering step.

Run this from Command Prompt (not RStudio/reticulate) -- a fresh process
gives a clean memory slate, and this step alone can take a while given
Market Depth's size.

Usage:
    python run_option_b.py
"""

from pathlib import Path
import sys
import time

import merge_tx_md as mtm
import taq_stage2_ordering_and_cleaning as s2
import taq_stage3_merge as s3
import stage4_clean_classify as s4

BASE = Path(r"C:\Users\Lungelo Phewa\Documents\Data Science\Data Analysis for Trading\Assignment1")

TX_TRIMMED = BASE / "AGL_GRT_TX_trimmed.csv"
MD_FLAT = BASE / "AGL_GRT_MD_flat.parquet"          # un-filtered, confirmed to ~16:30 local
MERGED_V2 = BASE / "AGL_GRT_merged_v2.parquet"
CHUNKS_DIR_V2 = BASE / "taq_stage2_chunks_v2"
MERGED_SORTED_V2 = BASE / "AGL_GRT_merged_sorted_v2.parquet"
STAGE4_OUTPUT_V2 = BASE / "AGL_GRT_stage4_final_v2.parquet"


def step(name: str):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)


def main() -> int:
    overall_start = time.time()

    # ---- Step 1: merge_tx_md with the un-filtered MD file ----
    step("STEP 1/4: Merging TX + un-filtered MD")
    if not TX_TRIMMED.exists():
        print(f"ERROR: {TX_TRIMMED} not found."); return 1
    if not MD_FLAT.exists():
        print(f"ERROR: {MD_FLAT} not found."); return 1

    t0 = time.time()
    mtm.merge_tx_md(tx_path=TX_TRIMMED, md_path=MD_FLAT, output_path=MERGED_V2)
    print(f"Step 1 done in {time.time()-t0:.1f}s -> {MERGED_V2}")

    # ---- Step 2: Stage 2 (sort into chunks) ----
    step("STEP 2/4: Stage 2 -- sorting into chunks")
    CHUNKS_DIR_V2.mkdir(exist_ok=True)
    t0 = time.time()
    chunks_v2 = s2.create_sorted_chunks(
        input_path=MERGED_V2, temp_dir=CHUNKS_DIR_V2, chunk_size=s2.CHUNK_SIZE,
    )
    print(f"Step 2 done in {time.time()-t0:.1f}s -> {len(chunks_v2)} chunks")

    # ---- Step 3: Stage 3 (k-way merge) ----
    step("STEP 3/4: Stage 3 -- k-way merge (this is the slow one, ~30-40 min expected)")
    t0 = time.time()
    stats3 = s3.merge_sorted_chunks(chunks=chunks_v2, output_path=MERGED_SORTED_V2)
    print(f"Step 3 done in {time.time()-t0:.1f}s: {stats3}")

    ok = s3.verify_sorted(MERGED_SORTED_V2)
    print("verify_sorted() ->", ok)
    if not ok:
        print("ERROR: sort verification failed -- stopping before Stage 4.")
        return 1

    # ---- Step 4: Stage 4 (clean + classify) ----
    step("STEP 4/4: Stage 4 -- clean & classify")
    t0 = time.time()
    final_v2, stats4 = s4.run(input_path=MERGED_SORTED_V2, output_path=STAGE4_OUTPUT_V2)
    print(f"Step 4 done in {time.time()-t0:.1f}s")
    print(stats4)

    # ---- Verification: does the MD cutoff issue actually resolve? ----
    step("VERIFICATION: hourly event counts (expect quote coverage extending "
         "through ~hour 14 UTC now, not collapsing at hour 12)")
    final_v2["hour_utc"] = final_v2["ts_utc"].dt.hour
    print(final_v2.groupby(["hour_utc", "event_type"]).size().unstack("event_type").fillna(0).to_string())

    print()
    print(f"TOTAL elapsed: {(time.time()-overall_start)/60:.1f} minutes")
    print(f"Final output: {STAGE4_OUTPUT_V2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
