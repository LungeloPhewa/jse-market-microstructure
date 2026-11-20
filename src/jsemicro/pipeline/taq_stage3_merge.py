r"""
taq_stage3_merge.py
--------------------
Stage 3 of the TAQ data-work pipeline.

Takes the sorted parquet chunks produced by Stage 2
(`taq_stage2_ordering_and_cleaning.create_sorted_chunks`) and performs a
memory-bounded k-way merge to produce a single globally-sorted parquet
file, ordered by SORT_KEY.

This is deliberately written as a *streaming* external merge (Option B):
only one batch per chunk is ever held in memory at a time, regardless of
how many chunks or how many total rows there are. This is what lets the
approach scale from a 2-ticker test run up to the full 10-ticker, 6-month
dataset without blowing out RAM.

Usage from a Quarto / notebook cell (mirrors the Stage 2 call pattern):

    from pathlib import Path
    import taq_stage3_merge as m

    chunks = sorted(temp_dir.glob("sorted_chunk_*.parquet"))
    output_path = Path(r"C:\...\AGL_GRT_merged_sorted.parquet")

    stats = m.merge_sorted_chunks(
        chunks=chunks,
        output_path=output_path,
        sort_key=m.SORT_KEY,
        batch_size=m.MERGE_BATCH_SIZE,
    )

Standalone CLI usage:

    python taq_stage3_merge.py --chunks-dir <dir> --output <file.parquet>
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# Config — keep this in sync with taq_stage2_ordering_and_cleaning.py
# --------------------------------------------------------------------------

SORT_KEY = ["ric", "ts_utc", "SEQNUM", "event_id"]

# Requested per-chunk read batch size (upper bound). The ACTUAL per-chunk
# batch size used is auto-scaled down (see _resolve_batch_size) so that
# total_buffered_rows = n_chunks * effective_batch_size stays under
# MAX_TOTAL_BUFFERED_ROWS, regardless of how many chunks you're merging.
MERGE_BATCH_SIZE = 50_000

# Memory safety cap: total rows buffered across ALL chunk cursors at once,
# combined, will not exceed roughly this many rows. This is what actually
# bounds peak memory -- raise it if you have RAM to spare and want fewer,
# larger disk reads; lower it if you're still seeing memory pressure.
MAX_TOTAL_BUFFERED_ROWS = 200_000

# Rows accumulated in the output buffer before flushing to the parquet
# writer. Independent of the per-chunk read batch size above.
WRITE_BATCH_SIZE = 50_000

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --------------------------------------------------------------------------
# Chunk cursor: streams one sorted parquet file batch-by-batch
# --------------------------------------------------------------------------

def _resolve_batch_size(n_chunks: int, requested_batch_size: int) -> int:
    """
    Auto-scale the per-chunk read batch size so that, combined across ALL
    chunk cursors, total buffered rows stays under MAX_TOTAL_BUFFERED_ROWS.
    This is what actually keeps peak memory bounded when merging many
    chunks (e.g. 94), regardless of what batch_size the caller requested.
    """
    auto_cap = max(200, MAX_TOTAL_BUFFERED_ROWS // max(n_chunks, 1))
    effective = min(requested_batch_size, auto_cap)
    if effective < requested_batch_size:
        logger.info(
            "Scaling down per-chunk batch size from %s to %s "
            "(%d chunks x %s rows would exceed the %s-row memory cap)",
            f"{requested_batch_size:,}", f"{effective:,}", n_chunks,
            f"{requested_batch_size:,}", f"{MAX_TOTAL_BUFFERED_ROWS:,}",
        )
    return effective


def _scan_canonical_schema(chunks: list[Path]) -> pa.Schema:
    """
    Read only the METADATA (footer) of every chunk file -- no row data --
    and build one canonical pyarrow schema across all of them.

    This exists because a column that is entirely null within a given
    100,000-row chunk gets inferred by pyarrow as type `null` for that
    file, while the same column in a chunk that has real values gets a
    concrete type (e.g. `large_string`). Writing both into the same
    output parquet file back-to-back then fails with a schema mismatch.
    Here we prefer the concrete type over `null` wherever they conflict,
    so every batch gets cast to one consistent schema before writing.

    Raises a clear error if two chunks disagree on a genuinely different
    concrete type for the same column (a real data problem, not a
    null-inference artifact).
    """
    field_types: dict[str, pa.DataType] = {}
    field_order: list[str] = []

    for c in chunks:
        schema = pq.ParquetFile(str(c)).schema_arrow
        for f in schema:
            if f.name not in field_types:
                field_order.append(f.name)
                field_types[f.name] = f.type
                continue

            existing = field_types[f.name]
            if existing == f.type:
                continue
            if pa.types.is_null(existing) and not pa.types.is_null(f.type):
                field_types[f.name] = f.type  # upgrade null -> concrete type
            elif pa.types.is_null(f.type):
                pass  # keep existing concrete type, this chunk's is just null
            else:
                raise ValueError(
                    f"Column '{f.name}' has genuinely conflicting types across "
                    f"chunks: {existing} (seen earlier) vs {f.type} (in {c.name}). "
                    f"This needs manual investigation -- it isn't a null-inference artifact."
                )

    return pa.schema([pa.field(name, field_types[name]) for name in field_order])


def _sanitize_sort_key(key: tuple) -> tuple:
    """
    Make a sort key tuple safe for strict '<' comparisons (heapq, sorted
    comparisons, etc). Missing values -- NaN floats, NaT timestamps, None,
    pandas NA -- all violate a total order the same way NaN does (neither
    "<" nor ">" nor "==" behaves consistently), which silently corrupts
    heapq's internal invariant if one shows up in ANY sort key column, not
    just numeric ones. A NaT in ts_utc (the PRIMARY sort key) is actually
    more consequential than a NaN in a tie-break column like SEQNUM, and
    an earlier version of this function only caught plain NaN floats,
    missing NaT entirely.

    Uses pd.isna() (handles NaN, NaT, None, pd.NA uniformly) instead of a
    type-specific check. Every element becomes (0, value) normally, or
    (1, 0.0) if missing -- so missing values always compare as "after all
    real values" (matching pandas' own default sort behaviour), and the
    tuple comparison never has to compare a missing value against a real
    one directly.
    """
    sanitized = []
    for v in key:
        try:
            is_missing = bool(pd.isna(v))
        except (TypeError, ValueError):
            is_missing = False
        sanitized.append((1, 0.0) if is_missing else (0, v))
    return tuple(sanitized)


@dataclass
class _ChunkCursor:
    """
    Streams a single sorted chunk file one row at a time.

    Only ever holds ONE current (row, key) pair boxed in Python, plus the
    underlying pandas batch its lazy itertuples() generator is drawing
    from (bounded to `batch_size` rows, columnar / not boxed). This is
    what keeps memory O(n_chunks * batch_size) instead of blowing up when
    there are many chunks -- the earlier version materialized every row
    in the batch as a boxed tuple up front, for all chunks, before merging
    even started.

    `columns` MUST be the canonical column order (see _scan_canonical_schema)
    -- every batch read from this file is reindexed to that exact column
    order/set before being iterated, so row tuples always line up with the
    canonical schema regardless of this particular file's own on-disk
    column order or any columns it might be missing.
    """

    path: Path
    chunk_idx: int
    batch_size: int
    columns: list[str]
    key_indices: list[int] = field(default_factory=list, init=False)

    _pf: Any = field(default=None, init=False, repr=False)
    _batch_iter: Optional[Iterator[pa.RecordBatch]] = field(default=None, init=False, repr=False)
    _row_iter: Iterator[tuple] = field(default=iter(()), init=False, repr=False)
    _current: Optional[tuple] = field(default=None, init=False, repr=False)
    _current_key: Optional[tuple] = field(default=None, init=False, repr=False)
    exhausted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._pf = pq.ParquetFile(str(self.path))
        self.key_indices = [self.columns.index(k) for k in SORT_KEY]
        self._batch_iter = self._pf.iter_batches(batch_size=self.batch_size)
        self._row_iter = iter(())
        self._advance_row()

    def _advance_row(self) -> None:
        while True:
            try:
                record = next(self._row_iter)
            except StopIteration:
                try:
                    batch = next(self._batch_iter)
                except StopIteration:
                    self.exhausted = True
                    self._current = None
                    self._current_key = None
                    return
                df = batch.to_pandas()
                # Force consistent column order/set across every chunk file,
                # regardless of this file's own on-disk order. Missing
                # columns (shouldn't normally happen) become NaN rather
                # than raising.
                df = df.reindex(columns=self.columns)
                # Lazy generator: does NOT materialize all rows up front.
                self._row_iter = df.itertuples(index=False, name=None)
                continue
            self._current = record
            self._current_key = _sanitize_sort_key(tuple(record[i] for i in self.key_indices))
            return

    def peek_key(self) -> Optional[tuple]:
        return self._current_key

    def pop(self) -> tuple:
        """Return current record and advance, refilling from disk if needed."""
        record = self._current
        self._advance_row()
        return record


# --------------------------------------------------------------------------
# Output writer
# --------------------------------------------------------------------------

class _BufferedParquetWriter:
    """
    Buffers output rows and flushes to a ParquetWriter in fixed-size batches.

    Always casts to a single fixed `schema` (pyarrow Schema), rather than
    inferring a fresh schema from each batch's DataFrame -- this is what
    prevents the null-vs-concrete-type schema drift described in
    _scan_canonical_schema from ever reaching the writer.
    """

    def __init__(self, output_path: Path, schema: pa.Schema, batch_size: int):
        self.output_path = output_path
        self.schema = schema
        self.columns = schema.names
        self.batch_size = batch_size
        self._buffer: list[tuple] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self.rows_written = 0

    def add(self, record: tuple) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer, columns=self.columns)
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(str(self.output_path), self.schema)
        self._writer.write_table(table)
        self.rows_written += len(self._buffer)
        self._buffer = []

    def close(self) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()


# --------------------------------------------------------------------------
# Main merge routine
# --------------------------------------------------------------------------

def merge_sorted_chunks(
    chunks: list[Path],
    output_path: Path,
    sort_key: Optional[list[str]] = None,
    batch_size: int = MERGE_BATCH_SIZE,
    write_batch_size: int = WRITE_BATCH_SIZE,
    max_total_buffered_rows: int = MAX_TOTAL_BUFFERED_ROWS,
    log_path: Optional[Path] = None,
    progress_path: Optional[Path] = None,
    progress_every: int = 250_000,
) -> dict[str, Any]:
    """
    K-way merge a list of pre-sorted parquet chunk files into one globally
    sorted parquet file at `output_path`.

    Each input chunk MUST already be sorted by `sort_key` (this is what
    Stage 2's create_sorted_chunks guarantees). This function does not
    re-sort within a chunk — it only merges across chunks.

    Memory safety: the requested `batch_size` (rows read per chunk per
    disk hit) is auto-scaled DOWN so that n_chunks * effective_batch_size
    stays under `max_total_buffered_rows`. With many chunks (e.g. 94),
    this matters a lot — see _resolve_batch_size.

    Crash visibility: this writes to a persistent log FILE (not just
    the console) and periodically writes a small JSON progress file, so
    that if the console/GUI freezes or the process is killed, you can
    open the log/progress file afterward and see exactly how far the
    merge got and how long it had been running. This matters a lot when
    running inside RStudio/Quarto via reticulate, where a killed R
    session can wipe the console with no Python traceback at all.

    log_path: defaults to `<output_path>.stage3.log` next to the output.
    progress_path: defaults to `<output_path>.stage3_progress.json`.
    progress_every: rows between progress-file updates (also flushed to
        disk immediately, not buffered by the OS).

    Returns a small dict of stats for logging / sanity-checking in Quarto.
    """
    global SORT_KEY, MAX_TOTAL_BUFFERED_ROWS
    if sort_key is not None:
        SORT_KEY = sort_key
    MAX_TOTAL_BUFFERED_ROWS = max_total_buffered_rows

    if not chunks:
        raise ValueError("No chunks provided to merge_sorted_chunks().")

    output_path = Path(output_path)
    if log_path is None:
        log_path = output_path.with_suffix(output_path.suffix + ".stage3.log")
    if progress_path is None:
        progress_path = output_path.with_suffix(output_path.suffix + ".stage3_progress.json")

    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(file_handler)
    logger.info("=== Stage 3 merge starting. Log file: %s | Progress file: %s ===", log_path, progress_path)

    def _write_progress(total_rows: int, n_chunks_total: int, elapsed: float, done: bool = False) -> None:
        payload = {
            "total_rows_merged": total_rows,
            "n_chunks": n_chunks_total,
            "elapsed_seconds": round(elapsed, 1),
            "rows_per_second": round(total_rows / elapsed, 1) if elapsed > 0 else None,
            "done": done,
            "output_path": str(output_path),
        }
        tmp = progress_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(progress_path)  # atomic on Windows too

    try:
        chunks = sorted(chunks)  # deterministic order, cosmetic only
        effective_batch_size = _resolve_batch_size(len(chunks), batch_size)

        logger.info("Starting k-way merge of %d chunks using sort key: %s", len(chunks), SORT_KEY)
        logger.info(
            "Per-chunk read batch size: %s (requested %s) | write batch size: %s | memory cap: %s rows",
            f"{effective_batch_size:,}", f"{batch_size:,}", f"{write_batch_size:,}",
            f"{MAX_TOTAL_BUFFERED_ROWS:,}",
        )

        start = time.time()
        _write_progress(0, len(chunks), 0.0)

        logger.info("Scanning schemas across all %d chunks to build a canonical output schema...", len(chunks))
        canonical_schema = _scan_canonical_schema(chunks)
        canonical_columns = canonical_schema.names
        logger.info("Canonical schema resolved: %d columns.", len(canonical_columns))

        cursors = [
            _ChunkCursor(path=c, chunk_idx=i, batch_size=effective_batch_size, columns=canonical_columns)
            for i, c in enumerate(chunks)
        ]

        writer = _BufferedParquetWriter(output_path, schema=canonical_schema, batch_size=write_batch_size)

        # Heap entries: (key_tuple, chunk_idx) -> chunk_idx breaks ties deterministically
        heap: list[tuple[tuple, int]] = []
        for cur in cursors:
            key = cur.peek_key()
            if key is not None:
                heapq.heappush(heap, (key, cur.chunk_idx))

        cursors_by_idx = {c.chunk_idx: c for c in cursors}
        total_rows = 0
        last_log_rows = 0
        LOG_EVERY = min(progress_every, 250_000)

        while heap:
            _key, idx = heapq.heappop(heap)
            cur = cursors_by_idx[idx]
            record = cur.pop()
            writer.add(record)
            total_rows += 1

            next_key = cur.peek_key()
            if next_key is not None:
                heapq.heappush(heap, (next_key, idx))

            if total_rows - last_log_rows >= LOG_EVERY:
                elapsed_so_far = time.time() - start
                logger.info(
                    "Merged %s rows so far... (%.1fs elapsed, %.0f rows/sec)",
                    f"{total_rows:,}", elapsed_so_far,
                    total_rows / elapsed_so_far if elapsed_so_far > 0 else 0,
                )
                last_log_rows = total_rows

            if total_rows % progress_every == 0:
                _write_progress(total_rows, len(chunks), time.time() - start)

        writer.close()
        elapsed = time.time() - start

        stats = {
            "n_chunks": len(chunks),
            "total_rows": total_rows,
            "output_path": str(output_path),
            "elapsed_seconds": round(elapsed, 1),
        }

        logger.info(
            "Finished merge: %s rows from %d chunks -> %s (%.1fs)",
            f"{total_rows:,}", len(chunks), output_path.name, elapsed,
        )
        _write_progress(total_rows, len(chunks), elapsed, done=True)

        return stats

    except Exception:
        logger.exception("Stage 3 merge FAILED. See traceback above / in log file %s", log_path)
        raise
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def verify_sorted(output_path: Path, sort_key: Optional[list[str]] = None, sample: Optional[int] = None) -> bool:
    """
    Lightweight sanity check: confirms the merged file is non-decreasing
    on sort_key. Reads in batches so it stays memory-bounded on huge files.
    Set `sample` to an integer to only check the first N rows (fast pre-check).

    Keys are run through _sanitize_sort_key before comparison, same as
    during the merge itself, so a NaN in a tie-break column (e.g. SEQNUM)
    can't produce a misleading result here either.
    """
    key = sort_key or SORT_KEY
    pf = pq.ParquetFile(str(output_path))
    prev_key = None
    prev_raw = None
    n_checked = 0
    for batch in pf.iter_batches(batch_size=100_000):
        df = batch.to_pandas()
        raw_keys = list(df[key].itertuples(index=False, name=None))
        keys = [_sanitize_sort_key(k) for k in raw_keys]
        if prev_key is not None and keys and keys[0] < prev_key:
            logger.error("Order violation at row %d: %s < %s", n_checked, raw_keys[0], prev_raw)
            return False
        for i in range(1, len(keys)):
            if keys[i] < keys[i - 1]:
                logger.error("Order violation at row %d: %s < %s", n_checked + i, raw_keys[i], raw_keys[i - 1])
                return False
        if keys:
            prev_key = keys[-1]
            prev_raw = raw_keys[-1]
        n_checked += len(keys)
        if sample is not None and n_checked >= sample:
            break
    logger.info("Verified sort order over %s rows: OK", f"{n_checked:,}")
    return True


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: k-way merge sorted TAQ chunks.")
    parser.add_argument("--chunks-dir", type=Path, required=True, help="Directory containing sorted_chunk_*.parquet files")
    parser.add_argument("--pattern", type=str, default="sorted_chunk_*.parquet")
    parser.add_argument("--output", type=Path, required=True, help="Path for the merged output parquet file")
    parser.add_argument("--batch-size", type=int, default=MERGE_BATCH_SIZE)
    parser.add_argument("--verify", action="store_true", help="Run a full sort-order verification after merging")
    args = parser.parse_args()

    chunks = sorted(args.chunks_dir.glob(args.pattern))
    if not chunks:
        raise SystemExit(f"No chunks matching {args.pattern} found in {args.chunks_dir}")

    stats = merge_sorted_chunks(chunks=chunks, output_path=args.output, batch_size=args.batch_size)
    print(stats)

    if args.verify:
        verify_sorted(args.output)


if __name__ == "__main__":
    main()
