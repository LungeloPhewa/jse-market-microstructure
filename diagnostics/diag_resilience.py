# Run from the console after the report's setup chunks, with:
#     exec(open(BASE / "diag_resilience.py").read())
# Needs: trades, md, res, RICS, SAST, pd (all defined by the report).

pd.set_option("display.width", 200)


def _local(ts):
    return ts.dt.tz_convert(SAST)


print("=" * 70)
print("1. Depth-feed updates around each pre-close candidate")
print("=" * 70)
c = res["candidates"].query("window == 'pre_close'")
for _, r in c.iterrows():
    t = r["shock_ts"]
    day = r["shock_local"].date()
    q = md.loc[(md["ric"] == r["ric"]) & (_local(md["ts"]).dt.date == day), "ts"]
    before, after = q[q <= t], q[q > t]
    gb = (t - before.max()).total_seconds() if len(before) else float("nan")
    ga = (after.min() - t).total_seconds() if len(after) else float("nan")
    print(f"{r['ric']} {r['shock_local']:%Y-%m-%d %H:%M:%S}  vol={r['shock_volume']:>12,.0f}  "
          f"quotes that day={len(q):>7,}  gap before={gb:>6.0f}s  gap after={ga:>6.0f}s")

print()
print("=" * 70)
print("2. Activity per 10-minute bucket from 14:00 SAST (median across days)")
print("   quote_upd = depth-feed updates, trades = trade prints")
print("=" * 70)
for ric in RICS:
    q = _local(md.loc[md["ric"] == ric, "ts"])
    t = _local(trades.loc[trades["ric"] == ric, "ts"])
    def per_bucket(s):
        df = pd.DataFrame({"d": s.dt.date, "b": s.dt.floor("10min").dt.strftime("%H:%M")})
        df = df[df["b"] >= "14:00"]
        return df.groupby(["d", "b"]).size().unstack("b", fill_value=0).median()
    tab = pd.DataFrame({"quote_upd": per_bucket(q), "trades": per_bucket(t)}).fillna(0)
    print(f"\n{ric}\n{tab.astype(int).T.to_string()}")

print()
print("=" * 70)
print("3. Last depth update vs last trade per day (SAST)")
print("=" * 70)
for ric in RICS:
    q = _local(md.loc[md["ric"] == ric, "ts"])
    t = _local(trades.loc[trades["ric"] == ric, "ts"])
    # second-to-last update shows whether 16:30 is a lone end-of-day stamp
    def nth_last(s, n):
        return s.groupby(s.dt.date).apply(lambda x: x.sort_values().iloc[-n] if len(x) >= n else pd.NaT)
    tab = pd.DataFrame({
        "last_quote": q.groupby(q.dt.date).max().dt.strftime("%H:%M:%S"),
        "2nd_last_quote": nth_last(q, 2).dt.strftime("%H:%M:%S"),
        "last_trade": t.groupby(t.dt.date).max().dt.strftime("%H:%M:%S"),
    })
    print(f"\n{ric}\n{tab.to_string()}")

print()
print("=" * 70)
print("4. Feed alignment: share of trades priced inside the prevailing bid-ask")
print("   (the correct offset should stand out clearly)")
print("=" * 70)
for ric in RICS:
    tx = trades.loc[trades["ric"] == ric, ["ts", "price"]].sort_values("ts")
    qq = md.loc[md["ric"] == ric, ["ts", "bid", "ask"]].sort_values("ts")
    qq = qq[(qq["bid"] > 0) & (qq["ask"] > qq["bid"])]
    res_off = {}
    for off in ["-2h", "-1h", "0h", "1h", "2h"]:
        m = pd.merge_asof(tx.assign(ts=tx["ts"] + pd.Timedelta(off)), qq, on="ts")
        res_off[off] = ((m["price"] >= m["bid"]) & (m["price"] <= m["ask"])).mean()
    print(ric, "  ".join(f"{k:>3}: {v:6.1%}" for k, v in res_off.items()))
