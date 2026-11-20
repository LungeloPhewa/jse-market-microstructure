"""
build_daily_auction_prices.py
--------------------------------
Assignment step 1.2: daily opening/closing AUCTION prices (not last
continuous-session tick), derived directly from the raw Transactions
tape rather than a separate Bloomberg/Refinitiv daily pull -- the
dataset only includes intraday Transactions + Market
Depth files, no separate daily/EOD file.

HOW AUCTION PRINTS ARE IDENTIFIED
------------------------------------
Investigation of the raw FID universe (all ~370 possible event
attributes, most of which trim_tx_columns.py discards) found:

  - ACT_FLAG1 == 'U' marks a genuine auction UNCROSS print. Confirmed by
    checking every AGL/GRT event with this flag across the full sample:
    TRDVOL_1 always exactly equals AUCTIONVOL for these rows, and they
    occur almost exclusively at ~07:00 GMT (09:00 SAST, market open) or
    ~15:00 GMT (17:00 SAST, market close) -- exactly matching the JSE's
    trading calendar.
  - AUCTIONVOL alone is NOT a reliable filter on its own: it's a
    "sticky" field that carries the last-known auction volume forward
    onto unrelated subsequent VERIFY_SYNC_PC messages. ACT_FLAG1=='U' is
    the precise marker; those carried-forward rows all show ACT_FLAG1
    as blank/space, not 'U'.
  - A handful of dates show a THIRD 'U'-flagged event mid-session --
    most likely a JSE volatility/circuit-breaker reopening auction
    (one such date directly coincides with an unusually large trade
    already flagged as a volume outlier during Stage 5 EDA). These are
    kept but excluded from the primary open/close columns, logged
    separately.

OUTPUT
------
One row per (ric, date):
    ric, date,
    open_auction_price, open_auction_volume, open_auction_time_utc,
    close_auction_price, close_auction_volume, close_auction_time_utc,
    n_auction_events (total 'U'-flagged events that day, expect 2),
    n_extra_intraday_auctions (0 normally; >0 flags a reopening auction)
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TARGET_RICS = {"AGLJ.J", "GRTJ.J"}
JSE_OFFSET_HOURS = 2  # GMT+2 (SAST), matches flatten_taq.py's JSE_FIXED_OFFSET_HOURS

# Column positions in the raw file's header row (0-indexed), per the
# confirmed schema in flatten_taq.py's module docstring.
COL_RIC = 0
COL_DATE = 1
COL_TIME = 2
COL_TYPE = 4
COL_MSG_TYPE = 6
COL_FID_NAME = 7
COL_FID_VALUE = 8


def extract_auction_events(raw_path: str | Path) -> pd.DataFrame:
    """
    Single streaming pass over the raw (decompressed) Transactions CSV,
    reconstructing one event at a time (header + its FID rows) and
    keeping only AGL/GRT events flagged ACT_FLAG1 == 'U'.

    Memory-bounded: never holds more than one event's FIDs in memory at
    a time, regardless of file size (this raw file is ~1GB uncompressed).
    """
    events: list[dict] = []
    current_header: list[str] | None = None
    current_fids: dict[str, str] = {}

    def flush() -> None:
        if current_header is None:
            return
        ric = current_header[COL_RIC]
        if ric not in TARGET_RICS:
            return
        if current_fids.get("ACT_FLAG1") != "U":
            return
        events.append({
            "ric": ric,
            "date_raw": current_header[COL_DATE],
            "time_raw": current_header[COL_TIME],
            "msg_type": current_header[COL_MSG_TYPE],
            "TRDPRC_1": current_fids.get("TRDPRC_1"),
            "TRDVOL_1": current_fids.get("TRDVOL_1"),
            "AUCTIONVOL": current_fids.get("AUCTIONVOL"),
        })

    raw_path = Path(raw_path)
    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rt", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        next(reader)  # header row
        for row in reader:
            if row[COL_TYPE] != "FID":
                flush()
                current_header = row
                current_fids = {}
            else:
                current_fids[row[COL_FID_NAME]] = row[COL_FID_VALUE]
        flush()

    df = pd.DataFrame(events)
    log.info("Found %d auction-flagged (ACT_FLAG1='U') events across %s.",
              len(df), TARGET_RICS)

    # Date[G] / Time[G] are GMT: they ARE the UTC instant (see flatten_taq.py).
    # The original Assignment 1 version subtracted the +2h offset here, which
    # shifted every auction two hours early; the calendar date is taken in
    # SAST after the conversion.
    combined = df["date_raw"] + " " + df["time_raw"]
    naive = pd.to_datetime(combined, format="%d-%b-%Y %H:%M:%S.%f", errors="coerce")
    naive = naive.fillna(pd.to_datetime(combined, format="%d-%b-%Y %H:%M:%S", errors="coerce"))
    df["ts_utc"] = naive.dt.tz_localize("UTC")
    df["date"] = df["ts_utc"].dt.tz_convert("Africa/Johannesburg").dt.date

    df["TRDPRC_1"] = pd.to_numeric(df["TRDPRC_1"], errors="coerce")
    df["TRDVOL_1"] = pd.to_numeric(df["TRDVOL_1"], errors="coerce")
    df["AUCTIONVOL"] = pd.to_numeric(df["AUCTIONVOL"], errors="coerce")

    return df.sort_values(["ric", "ts_utc"]).reset_index(drop=True)


def extract_auction_events_from_flat(tx_flat_path: str | Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Same as :func:`extract_auction_events`, but read from the flattened
    Transactions CSV written by ``flatten_taq.py`` (one row per event), which
    keeps ``ACT_FLAG1`` and ``AUCTIONVOL``. Lets the auction table be rebuilt
    when only the flattened file, not the raw export, is available.
    """
    cols = ["ric", "ts_utc", "message_type", "ACT_FLAG1", "TRDPRC_1", "TRDVOL_1", "AUCTIONVOL"]
    parts = []
    for ch in pd.read_csv(tx_flat_path, usecols=cols, chunksize=chunksize, dtype=str):
        keep = ch["ric"].isin(TARGET_RICS) & (ch["ACT_FLAG1"].str.strip() == "U")
        parts.append(ch.loc[keep])
    df = pd.concat(parts, ignore_index=True).rename(columns={"message_type": "msg_type"})
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    df["date"] = df["ts_utc"].dt.tz_convert("Africa/Johannesburg").dt.date
    for c in ("TRDPRC_1", "TRDVOL_1", "AUCTIONVOL"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log.info("Found %d auction-flagged (ACT_FLAG1='U') events in %s.", len(df), tx_flat_path)
    return df.sort_values(["ric", "ts_utc"]).reset_index(drop=True)


def build_daily_auction_table(auction_events: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the raw auction-event stream into one row per (ric, date):
    the EARLIEST 'U' event each day is the opening auction, the LATEST
    is the closing auction. Anything in between (rare) is a candidate
    intraday reopening auction, counted but excluded from open/close.

    A handful of (ric, date) pairs have only ONE 'U' event that day
    (observed for GRTJ.J, the less liquid of the two names, on 2 dates)
    -- almost certainly a genuinely missing auction print (e.g.
    insufficient crossing interest at the open for a quieter stock that
    day), not a processing error. Rather than duplicating that single
    event into both open AND close (misleading -- it would look like a
    real second print occurred), the single event is assigned to
    whichever side its own timestamp is closest to (before/after 09:00
    UTC, roughly the midpoint between the ~07:00 and ~15:00 GMT auction
    times), and the other side is left NaN.
    """
    rows = []
    for (ric, date), grp in auction_events.groupby(["ric", "date"]):
        grp = grp.sort_values("ts_utc")
        n_extra = max(0, len(grp) - 2)

        if len(grp) == 1:
            only_row = grp.iloc[0]
            is_opening = only_row["ts_utc"].hour < 9  # rough open/close midpoint
            rows.append({
                "ric": ric, "date": date,
                "open_auction_price": only_row["TRDPRC_1"] if is_opening else pd.NA,
                "open_auction_volume": only_row["TRDVOL_1"] if is_opening else pd.NA,
                "open_auction_time_utc": only_row["ts_utc"] if is_opening else pd.NaT,
                "close_auction_price": only_row["TRDPRC_1"] if not is_opening else pd.NA,
                "close_auction_volume": only_row["TRDVOL_1"] if not is_opening else pd.NA,
                "close_auction_time_utc": only_row["ts_utc"] if not is_opening else pd.NaT,
                "n_auction_events": 1,
                "n_extra_intraday_auctions": 0,
            })
            continue

        open_row = grp.iloc[0]
        close_row = grp.iloc[-1]
        rows.append({
            "ric": ric,
            "date": date,
            "open_auction_price": open_row["TRDPRC_1"],
            "open_auction_volume": open_row["TRDVOL_1"],
            "open_auction_time_utc": open_row["ts_utc"],
            "close_auction_price": close_row["TRDPRC_1"],
            "close_auction_volume": close_row["TRDVOL_1"],
            "close_auction_time_utc": close_row["ts_utc"],
            "n_auction_events": len(grp),
            "n_extra_intraday_auctions": n_extra,
        })

    daily = pd.DataFrame(rows).sort_values(["ric", "date"]).reset_index(drop=True)
    return daily


def run(tx_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Build the daily auction table from a raw export (``*.gz``) or from the
    flattened Transactions CSV, and write it to ``output_path`` (CSV)."""
    tx_path = Path(tx_path)
    if tx_path.suffix == ".gz":
        events = extract_auction_events(tx_path)
    else:
        events = extract_auction_events_from_flat(tx_path)
    daily = build_daily_auction_table(events)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)
    log.info("Wrote %d (ric, date) rows -> %s", len(daily), output_path)

    n_flagged_extra = int((daily["n_extra_intraday_auctions"] > 0).sum())
    if n_flagged_extra:
        log.warning(
            "%d (ric, date) pairs had extra intraday auction events "
            "(likely volatility/circuit-breaker reopening auctions):",
            n_flagged_extra,
        )
        print(daily.loc[daily["n_extra_intraday_auctions"] > 0,
                         ["ric", "date", "n_auction_events"]].to_string(index=False))

    expected_pairs = daily["ric"].nunique() * daily["date"].nunique()
    log.info("Coverage: %d (ric,date) rows found out of a max possible %d "
              "(gaps are expected on weekends/holidays -- cross-check against "
              "the 5 known low-activity dates from the intraday EDA).",
              len(daily), expected_pairs)

    return daily


if __name__ == "__main__":
    from ..config import PATHS
    run(PATHS.tx_flat, PATHS.auction_prices)
