"""
TAQ Pipeline -- Stage 2: Memory-Safe Deterministic Ordering + Cleaning

This version preserves the original Stage-2 intent while avoiding loading the
entire Parquet input into pandas at once.

Strategy
--------
1. Read the input Parquet file in row batches.
2. Sort each batch by the Stage-2 deterministic key and write temporary
   sorted Parquet chunks.
3. Perform a k-way merge of the sorted chunks using a bounded heap.
4. Apply the trading-hours filter and trade extraction while streaming the
   globally ordered rows into the two final Parquet outputs.

Important
---------
The chunked sort is an external merge sort. It therefore requires temporary
disk space, but memory usage is bounded by CHUNK_SIZE rows plus the output
buffers.

For a CSV input, pandas still reads the CSV in chunks and follows the same
external-sort path.
"""

from __future__ import annotations

import heapq
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

SORT_KEY_PRIORITY = ["ric", "ts_utc", "RTL", "Sequence Number", "SEQNUM", "event_id"]

JSE_SESSION_START = "09:00"
JSE_SESSION_END = "17:00"

TRADE_TYPE_VALUES = {"Trade", "TRADE", "trade"}

EXCLUDED_TRADE_MESSAGE_TYPES = {"VERIFY_SYNC_PC"}

# Memory/disk trade-off. Reduce this if RAM is still tight.
CHUNK_SIZE = 100_000

# Number of rows held before a final Parquet write.
OUTPUT_BATCH_SIZE = 50_000


# ----------------------------------------------------------------------------
# Step 1: Deterministic sort key
# ----------------------------------------------------------------------------

def resolve_available_sort_key(df: pd.DataFrame) -> list[str]:
    """Pick the available Stage-2 sort columns in priority order.

    event_id is always forced into the key because Stage 1 guarantees it is
    unique and it is the final deterministic tie-breaker.
    """
    available = [col for col in SORT_KEY_PRIORITY if col in df.columns]

    if "event_id" not in available:
        available.append("event_id")

    log.info("Using sort key: %s", available)
    return available


def _normalise_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Stage-2 timestamp columns are pandas datetimes."""
    for col in ("ts_utc", "ts_local"):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _sort_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Sort one chunk using the exact Stage-2 deterministic key."""
    sort_key = resolve_available_sort_key(df)
    return df.sort_values(
        sort_key,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


# ----------------------------------------------------------------------------
# External sorting
# ----------------------------------------------------------------------------

def _write_sorted_chunk(
    df: pd.DataFrame,
    chunk_path: Path,
) -> None:
    """Write one already-sorted pandas chunk to Parquet."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, chunk_path)


def _iter_input_chunks(
    input_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield input data in bounded-size pandas DataFrames."""
    if input_path.suffix.lower() == ".parquet":
        parquet_file = pq.ParquetFile(input_path)

        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            df = batch.to_pandas()
            yield _normalise_datetime_columns(df)

    elif input_path.suffix.lower() == ".csv":
        for df in pd.read_csv(
            input_path,
            parse_dates=["ts_utc", "ts_local"],
            chunksize=chunk_size,
        ):
            yield df

    else:
        raise ValueError(
            f"Unsupported input extension {input_path.suffix!r}; "
            "expected .parquet or .csv."
        )


def create_sorted_chunks(
    input_path: Path,
    temp_dir: Path,
    chunk_size: int = CHUNK_SIZE,
) -> list[Path]:
    """Create externally sorted Parquet chunks."""
    chunk_paths: list[Path] = []

    log.info(
        "Creating externally sorted chunks from %s (chunk size=%d rows)",
        input_path,
        chunk_size,
    )

    total_rows = 0

    for chunk_number, df in enumerate(
        _iter_input_chunks(input_path, chunk_size),
        start=1,
    ):
        if df.empty:
            continue

        total_rows += len(df)

        sorted_df = _sort_chunk(df)

        chunk_path = temp_dir / f"sorted_chunk_{chunk_number:06d}.parquet"
        _write_sorted_chunk(sorted_df, chunk_path)
        chunk_paths.append(chunk_path)

        log.info(
            "Created sorted chunk %d: %d rows (total read: %d)",
            chunk_number,
            len(sorted_df),
            total_rows,
        )

        # Explicitly release references before reading the next chunk.
        del sorted_df
        del df

    log.info(
        "Finished chunk sorting: %d rows across %d temporary chunks.",
        total_rows,
        len(chunk_paths),
    )

    return chunk_paths


# ----------------------------------------------------------------------------
# Streaming k-way merge
# ----------------------------------------------------------------------------

def _make_sort_value(value: Any) -> Any:
    """Convert pandas values into heap-comparable values.

    Missing values are placed after non-missing values, matching
    pandas sort_values(..., na_position='last').

    The tuple also carries a type tag so heterogeneous Python values do not
    accidentally become incomparable in the heap.
    """
    if pd.isna(value):
        return (1, 0)

    if isinstance(value, pd.Timestamp):
        return (0, value.value)

    if isinstance(value, (int, float)):
        return (0, value)

    return (0, str(value))


class _HeapItem:
    """Heap item for deterministic k-way merge."""

    __slots__ = ("key", "chunk_index", "row")

    def __init__(
        self,
        key: tuple,
        chunk_index: int,
        row: dict[str, Any],
    ):
        self.key = key
        self.chunk_index = chunk_index
        self.row = row

    def __lt__(self, other: "_HeapItem") -> bool:
        if self.key != other.key:
            return self.key < other.key
        return self.chunk_index < other.chunk_index


class _ChunkReader:
    """Small bounded-memory reader for one sorted Parquet chunk."""

    def __init__(self, path: Path, batch_size: int = OUTPUT_BATCH_SIZE):
        self.path = path
        self.file = pq.ParquetFile(path)
        self.batches = self.file.iter_batches(batch_size=batch_size)
        self.rows: list[dict[str, Any]] = []
        self.position = 0
        self._load_next_batch()

    def _load_next_batch(self) -> None:
        try:
            batch = next(self.batches)
        except StopIteration:
            self.rows = []
            self.position = 0
            return

        table = pa.Table.from_batches([batch])
        df = table.to_pandas()
        self.rows = df.to_dict("records")
        self.position = 0
        del df

    def next_row(self) -> dict[str, Any] | None:
        if self.position >= len(self.rows):
            self._load_next_batch()

        if not self.rows:
            return None

        row = self.rows[self.position]
        self.position += 1
        return row


def _row_sort_key(row: dict[str, Any], sort_key: list[str]) -> tuple:
    return tuple(_make_sort_value(row.get(col)) for col in sort_key)


def iter_merged_rows(
    chunk_paths: list[Path],
    sort_key: list[str],
) -> Iterator[dict[str, Any]]:
    """Yield rows from all sorted chunks in one globally sorted stream."""
    readers = [_ChunkReader(path) for path in chunk_paths]
    heap: list[_HeapItem] = []

    for i, reader in enumerate(readers):
        row = reader.next_row()
        if row is not None:
            heapq.heappush(
                heap,
                _HeapItem(_row_sort_key(row, sort_key), i, row),
            )

    while heap:
        item = heapq.heappop(heap)
        yield item.row

        next_row = readers[item.chunk_index].next_row()

        if next_row is not None:
            heapq.heappush(
                heap,
                _HeapItem(
                    _row_sort_key(next_row, sort_key),
                    item.chunk_index,
                    next_row,
                ),
            )


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

def diagnose_ordering_stream(
    rows: Iterator[dict[str, Any]],
    sort_key: list[str],
) -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
    """Diagnostic wrapper that checks ordering without storing the dataset.

    The input is already globally sorted by ``sort_key``. Duplicate keys
    are therefore detected by comparing each row with the immediately
    preceding row rather than retaining millions of keys in memory.

    RTL monotonicity is checked within each (ric, second) group.

    Rows are yielded unchanged.
    """

    state: dict[str, Any] = {
        "rows": 0,
        "full_key_duplicates": 0,
        "key_without_id_duplicates": 0,
        "rtl_violations": 0,
    }

    previous_full_key = None
    previous_non_id_key = None

    previous_rtl: dict[tuple, Any] = {}

    def _generator() -> Iterator[dict[str, Any]]:
        nonlocal previous_full_key
        nonlocal previous_non_id_key

        for row in rows:
            state["rows"] += 1

            # --------------------------------------------------------
            # Full sort-key duplicate detection
            # --------------------------------------------------------
            full_key = _row_sort_key(row, sort_key)

            if (
                previous_full_key is not None
                and full_key == previous_full_key
            ):
                state["full_key_duplicates"] += 1

            previous_full_key = full_key

            # --------------------------------------------------------
            # Duplicate detection ignoring event_id
            # --------------------------------------------------------
            key_without_id = tuple(
                _make_sort_value(row.get(col))
                for col in sort_key
                if col != "event_id"
            )

            if (
                previous_non_id_key is not None
                and key_without_id == previous_non_id_key
            ):
                state["key_without_id_duplicates"] += 1

            previous_non_id_key = key_without_id

            # --------------------------------------------------------
            # RTL monotonicity within (ric, second)
            # --------------------------------------------------------
            if "RTL" in row and "ts_utc" in row and "ric" in row:
                ts = row.get("ts_utc")
                rtl = row.get("RTL")

                if pd.notna(ts) and pd.notna(rtl):
                    second = pd.Timestamp(ts).floor("s")
                    group = (row.get("ric"), second)

                    rtl_num = pd.to_numeric(
                        rtl,
                        errors="coerce",
                    )

                    if pd.notna(rtl_num):
                        previous = previous_rtl.get(group)

                        if (
                            previous is not None
                            and rtl_num < previous
                        ):
                            state["rtl_violations"] += 1

                        previous_rtl[group] = rtl_num

            yield row

    return _generator(), state

def log_ordering_diagnostics(
    state: dict[str, Any],
    sort_key: list[str],
) -> None:
    """Log the streaming ordering diagnostics."""
    log.info(
        "Ties on %s (before event_id tiebreak): %d rows involved in a tie.",
        [c for c in sort_key if c != "event_id"],
        state["key_without_id_duplicates"],
    )

    if state["full_key_duplicates"] > 0:
        log.warning(
            "%d rows still share an IDENTICAL full sort key (%s) -- "
            "ordering is NOT fully deterministic. Investigate event_id "
            "generation in Stage 1.",
            state["full_key_duplicates"],
            sort_key,
        )
    else:
        log.info(
            "PASS: full sort key is unique across all %d rows -- "
            "no unresolved ties.",
            state["rows"],
        )

    if state["rtl_violations"] > 0:
        log.warning(
            "RTL is NOT non-decreasing within %d (RIC, second) groups -- "
            "investigate before trusting 'previous value' operations.",
            state["rtl_violations"],
        )
    else:
        log.info(
            "PASS: RTL is non-decreasing within every (RIC, second) group."
        )


# ----------------------------------------------------------------------------
# Calendar-time cleaning
# ----------------------------------------------------------------------------

def _keep_trading_hours(row: dict[str, Any]) -> bool:
    ts = row.get("ts_local")

    if ts is None or pd.isna(ts):
        return False

    ts = pd.Timestamp(ts)

    return (
        ts.weekday() < 5
        and ts.time() >= pd.Timestamp(JSE_SESSION_START).time()
        and ts.time() <= pd.Timestamp(JSE_SESSION_END).time()
    )


def _is_trade_row(
    row: dict[str, Any],
    trade_price_col: str = "TRDPRC_1",
    trade_volume_col: str = "TRDVOL_1",
    message_type_col: str = "message_type",
    exclude_message_types: set[str] = EXCLUDED_TRADE_MESSAGE_TYPES,
) -> bool:
    """Apply the original Stage-2 trade-identification logic to one row."""
    if trade_price_col in row:
        price = row.get(trade_price_col)
        is_trade = pd.notna(price) and price != 0

        if trade_volume_col in row:
            vol = row.get(trade_volume_col)
            is_trade = is_trade and pd.notna(vol) and vol != 0

        if (
            is_trade
            and exclude_message_types
            and message_type_col in row
            and row.get(message_type_col) in exclude_message_types
        ):
            return False

        return bool(is_trade)

    if "type" not in row:
        raise KeyError(
            "Expected a 'type' column to identify trade events -- "
            "check Stage 1 output."
        )

    return row.get("type") in TRADE_TYPE_VALUES


# ----------------------------------------------------------------------------
# Streaming Parquet output
# ----------------------------------------------------------------------------

class _ParquetStreamWriter:
    """Write pandas rows to Parquet in bounded batches."""

    def __init__(self, path: Path, schema: pa.Schema):
        self.path = path
        self.schema = schema
        self.writer = pq.ParquetWriter(
            path,
            schema,
            compression="snappy",
        )
        self.buffer: list[dict[str, Any]] = []
        self.rows_written = 0

    def write_row(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)

        if len(self.buffer) >= OUTPUT_BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        df = pd.DataFrame.from_records(self.buffer)

        # Align column order and preserve the original schema.
        df = df.reindex(columns=self.schema.names)

        table = pa.Table.from_pandas(
            df,
            schema=self.schema,
            preserve_index=False,
        )

        self.writer.write_table(table)
        self.rows_written += len(df)

        self.buffer.clear()
        del table
        del df

    def close(self) -> None:
        self.flush()
        self.writer.close()


# ----------------------------------------------------------------------------
# Main Stage-2 pipeline
# ----------------------------------------------------------------------------

def process_stage2(
    input_path: str | Path,
    all_events_parquet_path: str | Path,
    trades_only_parquet_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """Run memory-safe ordering + diagnostics + cleaning + Parquet export.

    The public API remains compatible with the original three required
    arguments. `chunk_size` is optional and can be reduced if RAM is limited.
    """
    input_path = Path(input_path)
    all_events_parquet_path = Path(all_events_parquet_path)
    trades_only_parquet_path = Path(trades_only_parquet_path)

    log.info("Loading flattened events from %s", input_path)

    all_events_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    trades_only_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="taq_stage2_",
        dir=str(all_events_parquet_path.parent),
    ) as tmp:
        temp_dir = Path(tmp)

        # ------------------------------------------------------------
        # Pass 1: externally sort
        # ------------------------------------------------------------
        chunk_paths = create_sorted_chunks(
            input_path=input_path,
            temp_dir=temp_dir,
            chunk_size=chunk_size,
        )

        if not chunk_paths:
            raise ValueError("Input contains no rows.")

        # Get the sort key from the first chunk.
        first_schema = pq.read_schema(chunk_paths[0])
        sort_key = [c for c in SORT_KEY_PRIORITY if c in first_schema.names]
        if "event_id" not in sort_key:
            sort_key.append("event_id")

        # Read the original schema from the first sorted chunk.
        schema = first_schema

        # ------------------------------------------------------------
        # Pass 2: globally merge sorted chunks and stream outputs
        # ------------------------------------------------------------
        merged_rows = iter_merged_rows(chunk_paths, sort_key)

        diagnostic_rows, diagnostic_state = diagnose_ordering_stream(
            merged_rows,
            sort_key,
        )

        all_writer = _ParquetStreamWriter(
            all_events_parquet_path,
            schema,
        )
        trades_writer = _ParquetStreamWriter(
            trades_only_parquet_path,
            schema,
        )

        total_seen = 0
        retained = 0
        trades = 0
        excluded_trades = 0

        try:
            for row in diagnostic_rows:
                total_seen += 1

                if not _keep_trading_hours(row):
                    continue

                retained += 1
                all_writer.write_row(row)

                is_trade = _is_trade_row(row)

                if is_trade:
                    # Count excluded message types separately for logging.
                    if (
                        "message_type" in row
                        and row.get("message_type") in EXCLUDED_TRADE_MESSAGE_TYPES
                    ):
                        excluded_trades += 1
                    else:
                        trades += 1
                        trades_writer.write_row(row)

        finally:
            all_writer.close()
            trades_writer.close()

        log_ordering_diagnostics(diagnostic_state, sort_key)

        log.info(
            "Trading-hours filter: %d / %d rows retained (%.1f%%).",
            retained,
            total_seen,
            100 * retained / max(total_seen, 1),
        )

        log.info(
            "Trade identification via TRDPRC_1/TRDVOL_1 "
            "(excluding %s): %d / %d retained rows are genuine trades "
            "(%.2f%%).",
            EXCLUDED_TRADE_MESSAGE_TYPES,
            trades,
            retained,
            100 * trades / max(retained, 1),
        )

        if excluded_trades:
            log.info(
                "Excluded %d trade-price/volume matches with excluded "
                "message types: %s.",
                excluded_trades,
                EXCLUDED_TRADE_MESSAGE_TYPES,
            )

        log.info(
            "Wrote %d cleaned events -> %s and %d trades -> %s",
            all_writer.rows_written,
            all_events_parquet_path,
            trades_writer.rows_written,
            trades_only_parquet_path,
        )


# ----------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    process_stage2(
        input_path="data/clean/AGL_GRT_merged.parquet",
        all_events_parquet_path="data/clean/AGL_all_events.parquet",
        trades_only_parquet_path="data/clean/AGL_trades_only.parquet",
    )
