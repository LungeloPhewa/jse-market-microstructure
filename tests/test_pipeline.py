"""Pipeline stages on toy inputs: external sort, Lee-Ready classification."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from jsemicro.pipeline import taq_stage2_ordering_and_cleaning as s2
from jsemicro.pipeline import taq_stage3_merge as s3
from jsemicro.pipeline import stage4_clean_classify as s4


def test_external_sort_matches_in_memory_sort(tmp_path):
    rng = np.random.default_rng(0)
    n = 5_000
    df = pd.DataFrame({
        "ric": rng.choice(["AGLJ.J", "GRTJ.J"], n),
        "ts_utc": pd.Timestamp("2013-09-03", tz="UTC") + pd.to_timedelta(rng.integers(0, 10**6, n), unit="ms"),
        "SEQNUM": rng.integers(0, 100, n).astype(float),
        "event_id": [f"{i:08x}" for i in range(n)],
        "x": rng.normal(size=n),
    })
    df.loc[rng.choice(n, 50, replace=False), "SEQNUM"] = np.nan          # missing keys sort last
    src = tmp_path / "in.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), src)
    chunks = s2.create_sorted_chunks(src, tmp_path, chunk_size=700)
    assert len(chunks) == 8
    out = tmp_path / "sorted.parquet"
    s3.merge_sorted_chunks(chunks, out, batch_size=100)
    got = pd.read_parquet(out)
    want = df.sort_values(["ric", "ts_utc", "SEQNUM", "event_id"], na_position="last", kind="mergesort")
    assert s3.verify_sorted(out)
    assert list(got["event_id"]) == list(want["event_id"])


def test_lee_ready_quote_rule_and_tick_test():
    df = pd.DataFrame({
        "ric": "AGLJ.J",
        "event_type": "trade",
        "TRDPRC_1": [101.0, 99.0, 100.0, 100.0, 102.0, 100.0],
        "mid_price": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    })
    out = s4.classify_trades(df)
    # above mid -> buy, below -> sell; at the mid, compare with the last
    # *different* price: 100 after 99 is an uptick (+1), and stays +1 after
    # another 100 (last different price is still 99); 100 after 102 is a downtick.
    assert list(out["trade_sign"]) == [1, -1, 1, 1, 1, -1]
