"""Assignment 3 pipeline: Step-0 cleaning, event identification and Hawkes
calibration for every trading day, with results cached for the book.

Run through the command line (``python -m jsemicro hawkes [RIC] [YYYY-MM-DD]``)
or Snakemake. Outputs in ``<data root>/out/hawkes/``:

* ``clean.pkl``                  markers, classified trades, signed trades, L1 quotes
* ``events_<RIC>.parquet``       the ten event types, every day
* ``event_diag_<RIC>.csv``       per-day share of trade volume matched to book depletions
* ``fit_<RIC>_<YYYY-MM-DD>.pkl`` calibrated Hawkes model and validation tests

Existing files are skipped, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import pickle
import time

import pandas as pd

from ..config import PATHS, RICS
from . import a3_data
from . import a3_model as am
from . import order_events as oe


def load_or_build_clean():
    """Step 0 for both securities, cached in ``clean.pkl``."""
    out = PATHS.hawkes_dir
    out.mkdir(parents=True, exist_ok=True)
    clean_path = out / "clean.pkl"
    if clean_path.exists():
        with open(clean_path, "rb") as fh:
            return pickle.load(fh)
    d = a3_data.load_clean(PATHS.tx_trimmed, PATHS.md_flat, RICS)
    with open(clean_path, "wb") as fh:
        pickle.dump(d, fh, protocol=4)
    return d


def main(rics=None, days=None):
    """Identify events and fit one Hawkes model per security and day."""
    rics = rics or RICS
    out = PATHS.hawkes_dir
    d = load_or_build_clean()
    all_days = sorted(d["trades"]["date"].unique())
    days = days or all_days

    def loader(ric, day):
        return a3_data.load_book_day(PATHS.md_flat, ric, day, d["markers"])

    for ric in rics:
        ev_path = out / f"events_{ric}.parquet"
        if ev_path.exists():
            events = pd.read_parquet(ev_path)
            events["date"] = pd.to_datetime(events["date"]).dt.date
        else:
            t0 = time.time()
            events, diag = oe.identify_events(d["trades"], loader, [ric], all_days)
            events.to_parquet(ev_path)
            diag.to_csv(out / f"event_diag_{ric}.csv", index=False)
            print(f"{ric}: {len(events):,} events in {time.time() - t0:.0f} s", flush=True)
        for day in days:
            fpath = out / f"fit_{ric}_{day}.pkl"
            if fpath.exists():
                continue
            t0 = time.time()
            fit = am.fit_day(events, d["markers"], ric, day)
            with open(fpath, "wb") as fh:
                pickle.dump(fit, fh, protocol=4)
            print(f"{ric} {day}: {fit['n_events']:,} events, fitted in {time.time() - t0:.0f} s", flush=True)
    (out / "DONE").write_text("ok\n")
