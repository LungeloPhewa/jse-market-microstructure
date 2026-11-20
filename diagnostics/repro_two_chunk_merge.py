"""
repro_two_chunk_merge.py
--------------------------
Fast, isolated reproduction of the merge violation using ONLY the two
chunk files actually involved (002 and 024, per diagnose_merge_violation.py's
output) plus 023 for completeness. This is small (a few hundred thousand
rows at most) so it runs in seconds, not 35 minutes -- lets us iterate on
the actual bug quickly instead of waiting on the full 94-chunk merge.

Usage from Command Prompt:

    python repro_two_chunk_merge.py
"""

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

import taq_stage3_merge as m

# --------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\Lungelo Phewa\Documents\Data Science\Data Analysis for Trading\Assignment1")
CHUNKS_DIR = BASE_DIR / "taq_stage2_chunks"
CHUNK_NAMES = ["sorted_chunk_000002.parquet", "sorted_chunk_000023.parquet", "sorted_chunk_000024.parquet"]
OUTPUT_PATH = BASE_DIR / "repro_two_chunk_merged.parquet"
# --------------------------------------------------------------------


def inspect_dtypes():
    print("=" * 78)
    print("DTYPE CHECK: sort-key columns across the involved chunks")
    print("=" * 78)
    for name in CHUNK_NAMES:
        path = CHUNKS_DIR / name
        df = pd.read_parquet(path, columns=m.SORT_KEY)
        print(f"\n{name}:")
        print(df.dtypes.to_string())
        # Also show the underlying pyarrow schema for these columns, which
        # is what actually determines the dtype after to_pandas().
        pf = pq.ParquetFile(str(path))
        arrow_types = {f.name: f.type for f in pf.schema_arrow if f.name in m.SORT_KEY}
        print("  Arrow types:", arrow_types)


def run_isolated_merge():
    print()
    print("=" * 78)
    print(f"Running merge_sorted_chunks() on just {CHUNK_NAMES}")
    print("=" * 78)

    chunks = [CHUNKS_DIR / name for name in CHUNK_NAMES]
    for c in chunks:
        if not c.exists():
            print(f"ERROR: {c} does not exist -- check CHUNK_NAMES / CHUNKS_DIR above.")
            return

    stats = m.merge_sorted_chunks(chunks=chunks, output_path=OUTPUT_PATH)
    print("Merge stats:", stats)

    ok = m.verify_sorted(OUTPUT_PATH)
    print("verify_sorted() ->", ok)

    if not ok:
        print()
        print("REPRODUCED with just these chunks. Good -- this is now a fast,")
        print("small repro we can debug quickly instead of the full 35-minute run.")
    else:
        print()
        print("Did NOT reproduce with just these 3 chunks in isolation.")
        print("This means a 4th chunk (or more) is needed to trigger it, or the")
        print("violation depends on interaction with additional chunks/cursors")
        print("being active in the heap at the same time.")


def main():
    inspect_dtypes()
    run_isolated_merge()


if __name__ == "__main__":
    main()
