"""Ingestion: flattening the FID format and the GMT -> SAST convention."""
import gzip

import pandas as pd
import pytest

from jsemicro.ingest import flatten_taq, auction_prices

HEADER = ("#RIC,Date[G],Time[G],GMT Offset,Type,MSG/FID Number,Message Type,FID Name,FID Value,"
          "FID Enum String,PE Code,Sequence Number,Source RIC,Number of FIDs\n")
ROWS = [
    "AGLJ.J,03-SEP-2013,07:00:09.236,+2,Raw-TAS,,UPDATE_PC,,,,,,,2",
    ",,,,FID,,,BID,24500,,,,,",
    ",,,,FID,,,BIDSIZE,300,,,,,",
    "AGLJ.J,03-SEP-2013,07:00:10,+2,Raw-TAS,,UPDATE_PC,,,,,,,3",      # no fractional seconds
    ",,,,FID,,,TRDPRC_1,24505,,,,,",
    ",,,,FID,,,TRDVOL_1,156,,,,,",
    ",,,,FID,,,ACT_FLAG1,U,,,,,",
    "GRTJ.J,03-SEP-2013,15:00:01.500,+2,Raw-TAS,,UPDATE_PC,,,,,,,1",
    ",,,,FID,,,TRDPRC_1,2400,,,,,",
]


@pytest.fixture
def raw_file(tmp_path):
    p = tmp_path / "raw.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(HEADER + "\n".join(ROWS) + "\n")
    return p


def test_flatten_one_row_per_event(raw_file, tmp_path):
    out = tmp_path / "flat.csv"
    flatten_taq.flatten_taq_file(raw_file, out, chunksize=4)      # chunks split events
    df = pd.read_csv(out)
    assert len(df) == 3
    assert {"BID", "BIDSIZE", "TRDPRC_1", "TRDVOL_1", "ACT_FLAG1"} <= set(df.columns)
    assert df.loc[1, "TRDPRC_1"] == 24505 and df.loc[1, "TRDVOL_1"] == 156
    assert df.loc[0, "BID"] == 24500 and pd.isna(df.loc[0, "TRDPRC_1"])
    assert df["event_id"].is_unique


def test_gmt_is_utc_and_sast_is_two_hours_later(raw_file, tmp_path):
    out = tmp_path / "flat.csv"
    flatten_taq.flatten_taq_file(raw_file, out)
    df = pd.read_csv(out)
    ts = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    local = pd.to_datetime(df["ts_local"], format="ISO8601", utc=True).dt.tz_convert("Africa/Johannesburg")
    assert ts.iloc[0] == pd.Timestamp("2013-09-03 07:00:09.236", tz="UTC")
    assert local.iloc[0].hour == 9                                 # 07:00 GMT = 09:00 SAST
    assert ts.iloc[1] == pd.Timestamp("2013-09-03 07:00:10", tz="UTC")


def test_wrong_gmt_offset_is_rejected(tmp_path):
    p = tmp_path / "bad.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(HEADER + ROWS[0].replace("+2", "+3") + "\n" + ROWS[1] + "\n")
    with pytest.raises(ValueError, match="GMT offsets"):
        flatten_taq.flatten_taq_file(p, tmp_path / "flat.csv")


def test_auction_events_from_raw(raw_file):
    ev = auction_prices.extract_auction_events(raw_file)
    assert len(ev) == 1
    assert ev.loc[0, "TRDPRC_1"] == 24505
    assert ev.loc[0, "ts_utc"] == pd.Timestamp("2013-09-03 07:00:10", tz="UTC")
