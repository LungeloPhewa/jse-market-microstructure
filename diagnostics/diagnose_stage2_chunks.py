"""
diagnose_stage2_chunks.py
--------------------------
Fast diagnostic: checks whether each individual Stage 2 sorted chunk file
is correctly sorted WITHIN ITSELF, independent of the Stage 3 k-way merge.

Why this matters: the k-way merge in taq_stage3_merge.py only merges
ACROSS chunk files -- it assumes each chunk is already sorted internally
(that's the whole external-merge-sort contract). If a chunk file has an
internal ordering bug, the merge will faithfully reproduce that disorder
in the final output no matter how correct the merge logic itself is.

This only reads the 4 sort-key columns (not all 60), and processes one
chunk file at a time, so it's fast (seconds, not minutes) even though the
full merge takes ~35 minutes.

Usage from Command Prompt:

    python diagnose_stage2_chunks.py
"""

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

import taq_stage3_merge as m

# --------------------------------------------------------------------
# EDIT IF NEEDED
# --------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\Lungelo Phewa\Documents\Data Science\Data Analysis for Trading\Assignment1")
CHUNKS_DIR = BASE_DIR / "taq_stage2_chunks"
# --------------------------------------------------------------------

SORT_KEY = m.SORT_KEY


def check_chunk(path: Path) -> list[dict]:
    """Return a list of violation dicts found within this single chunk file."""
    df = pd.read_parquet(path, columns=SORT_KEY)
    raw_keys = list(df.itertuples(index=False, name=None))
    keys = [m._sanitize_sort_key(k) for k in raw_keys]

    violations = []
    for i in range(1, len(keys)):
        if keys[i] < keys[i - 1]:
            violations.append({
                "chunk": path.name,
                "local_row": i,
                "prev": raw_keys[i - 1],
                "curr": raw_keys[i],
            })
    return violations


def main() -> None:
    chunks = sorted(CHUNKS_DIR.glob("sorted_chunk_*.parquet"))
    print(f"Checking {len(chunks)} chunk files in {CHUNKS_DIR} ...")
    print()

    total_violations = 0
    bad_chunks = []

    for i, c in enumerate(chunks):
        violations = check_chunk(c)
        if violations:
            bad_chunks.append(c.name)
            total_violations += len(violations)
            print(f"[{i+1}/{len(chunks)}] {c.name}: {len(violations)} internal violation(s)")
            for v in violations[:5]:  # cap detail printout per chunk
                print(f"    local row {v['local_row']}: {v['prev']}  ->  {v['curr']}  (WENT BACKWARDS)")
            if len(violations) > 5:
                print(f"    ... and {len(violations) - 5} more in this chunk")
        else:
            print(f"[{i+1}/{len(chunks)}] {c.name}: OK")

    print()
    print("=" * 70)
    if total_violations == 0:
        print("RESULT: All chunk files are correctly sorted internally.")
        print("-> The bug is NOT in Stage 2's per-chunk sort.")
        print("-> This means the disorder is genuinely introduced during the")
        print("   Stage 3 merge (or possibly reflects real, non-monotonic")
        print("   timestamps already present in the source data across")
        print("   different source pulls -- worth checking the 'source' /")
        print("   'type' columns at the violation point).")
    else:
        print(f"RESULT: Found {total_violations} internal violation(s) across "
              f"{len(bad_chunks)} chunk file(s):")
        for name in bad_chunks:
            print(f"  - {name}")
        print()
        print("-> This confirms the bug is in Stage 2 (or its input data), not")
        print("   in the Stage 3 merge logic. The k-way merge is faithfully")
        print("   reproducing disorder that already exists within these chunks.")
    print("=" * 70)


if __name__ == "__main__":
    main()
