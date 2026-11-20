"""Command-line entry point: ``python -m jsemicro <command>``.

Commands (each is one node of the Snakemake pipeline in ``workflow/Snakefile``)::

    synth      generate synthetic raw exports in data/synthetic/raw
    flatten    raw Transactions export -> clean/AGL_GRT_TX_flat.csv
    trim       flat Transactions -> clean/AGL_GRT_TX_trimmed.csv
    depth      raw Market Depth export -> clean/AGL_GRT_MD_flat.parquet
    auctions   daily opening/closing auction prices -> out/daily_auction_prices.csv
    merge      trimmed TX + depth -> clean/AGL_GRT_merged.parquet (two RICs only)
    sort       external merge sort -> clean/AGL_GRT_merged_sorted.parquet
    stage4     clean, compact and classify -> clean/AGL_GRT_stage4_final.parquet
    hawkes     Step 0, event identification and Hawkes fits -> out/hawkes/
    checksums  write SHA-256 checksums of raw and clean files -> <data root>/SHA256SUMS
    verify     check the files against <data root>/SHA256SUMS

The data root is chosen with the environment variable JSEMICRO_DATA
(``real`` by default, or ``synthetic``); see ``jsemicro.config``.
"""
from __future__ import annotations

import datetime as dt
import shutil
import sys
import warnings

import pandas as pd

from .config import PATHS, RICS, data_root


def _ensure_dirs():
    for p in (PATHS.raw_dir, PATHS.clean_dir, PATHS.out_dir):
        p.mkdir(parents=True, exist_ok=True)


def cmd_synth(args):
    from .synthetic import generator
    generator.write_raw_exports(PATHS.raw_dir)


def cmd_flatten(args):
    from .ingest import flatten_taq
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    flatten_taq.flatten_taq_file(PATHS.raw_tx, PATHS.tx_flat)


def cmd_trim(args):
    from .ingest import trim_tx_columns
    trim_tx_columns.trim_columns(PATHS.tx_flat, PATHS.tx_trimmed)


def cmd_depth(args):
    from .ingest import market_depth_prepare
    market_depth_prepare.prepare_market_depth_file(PATHS.raw_md, PATHS.md_flat)


def cmd_auctions(args):
    from .ingest import auction_prices
    src = PATHS.raw_tx if PATHS.raw_tx.exists() else PATHS.tx_flat
    auction_prices.run(src, PATHS.auction_prices)


def cmd_merge(args):
    from .pipeline import merge_tx_md
    merge_tx_md.merge_tx_md(PATHS.tx_trimmed, PATHS.md_flat, PATHS.merged, rics=RICS)


def cmd_sort(args):
    from .pipeline import taq_stage2_ordering_and_cleaning as s2
    from .pipeline import taq_stage3_merge as s3
    if PATHS.sort_chunks.exists():
        shutil.rmtree(PATHS.sort_chunks)
    PATHS.sort_chunks.mkdir(parents=True)
    chunks = s2.create_sorted_chunks(PATHS.merged, PATHS.sort_chunks, chunk_size=s2.CHUNK_SIZE)
    s3.merge_sorted_chunks(chunks=chunks, output_path=PATHS.merged_sorted, batch_size=s3.MERGE_BATCH_SIZE,
                           log_path=PATHS.clean_dir / "stage3_merge.log",
                           progress_path=PATHS.clean_dir / "stage3_progress.json")
    if not s3.verify_sorted(PATHS.merged_sorted):
        raise SystemExit("Merged file is not globally sorted")
    shutil.rmtree(PATHS.sort_chunks)


def cmd_stage4(args):
    from .pipeline import stage4_clean_classify as s4
    s4.run(PATHS.merged_sorted, PATHS.stage4)


def cmd_hawkes(args):
    from .hawkes import run
    rics = [a for a in args if a.endswith(".J")] or None
    days = [dt.date.fromisoformat(a) for a in args if a[:2] == "20"] or None
    run.main(rics, days)


def cmd_checksums(args):
    from .integrity import write_checksums
    path = write_checksums()
    print(path.read_text())


def cmd_verify(args):
    from .integrity import verify_checksums
    bad = verify_checksums()
    if bad:
        raise SystemExit("Checksum mismatch: " + ", ".join(bad))
    print("All checksums match.")


COMMANDS = {name[4:]: fn for name, fn in globals().items() if name.startswith("cmd_")}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 1
    _ensure_dirs()
    print(f"[jsemicro] data root: {data_root()}", flush=True)
    COMMANDS[argv[0]](argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
