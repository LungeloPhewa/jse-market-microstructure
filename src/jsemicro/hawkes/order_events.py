"""
order_events.py -- identify the ten order event types of Jericevich, Chang &
Gebbie (2021, Table 1) from trade and quote (TAQ) data with ten levels of
market depth.

    #   event                       side
    1   market order (MO)           buy   (buyer-initiated; consumes the ask)
    2   market order (MO)           sell  (seller-initiated; consumes the bid)
    3   aggressive limit order      bid
    4   aggressive limit order      ask
    5   passive limit order         bid
    6   passive limit order         ask
    7   aggressive cancellation     bid
    8   aggressive cancellation     ask
    9   passive cancellation        bid
    10  passive cancellation        ask

The data do not contain individual order messages, so events are inferred,
following section 3.1 of the paper:

* Market orders: continuous-trading, on-book trade prints with the same
  Lee-Ready sign and no more than 10 ms apart are ONE market order (an
  aggressive order that walks the book prints once per price level).
* Limit orders and cancellations: consecutive snapshots of the ten-level
  book are compared price by price on each side.
    - Quantity added at a price is a limit order. It is AGGRESSIVE if its
      price is at or better than the best price on its side before the
      update (it joins or improves the best, or the side was empty), and
      PASSIVE if worse.
    - Quantity removed at a price is either executed by a trade or
      cancelled. Removals are first matched against trade prints at that
      price, on that side, within +/- `match_window` of the update (the two
      feeds are time-stamped separately); whatever is not matched is a
      cancellation. It is AGGRESSIVE if the level was the best before the
      update, PASSIVE otherwise.
* Only prices visible in both snapshots are compared: when the book is ten
  levels deep, a level can enter or leave the ten-level window just because
  prices moved, which is not an order.
* Every changed price level counts as one event.

Only continuous-trading data are used (Step 0, market_phases.py).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

TZ_LOCAL = "Africa/Johannesburg"
EVENT_NAMES = {
    1: "Buy MO", 2: "Sell MO",
    3: "Aggressive bid LO", 4: "Aggressive ask LO",
    5: "Passive bid LO", 6: "Passive ask LO",
    7: "Aggressive bid cancel", 8: "Aggressive ask cancel",
    9: "Passive bid cancel", 10: "Passive ask cancel",
}
N_LEVELS = 10


def _epoch_ns(ts: pd.Series) -> np.ndarray:
    """Nanoseconds since the epoch, whatever the resolution of ``ts``.

    ``Series.astype("int64")`` returns the integer in the Series' own unit.
    pandas 3 parses timestamp strings (and pyarrow returns Parquet
    timestamps) at microsecond resolution, so a bare ``astype("int64")``
    gives microseconds, and the 10 ms / 500 ms windows below silently became
    10 s / 500 s. Converting to nanoseconds first makes the windows exact.
    """
    return ts.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()


def market_orders(trades, gap_ms=10.0):
    """Aggregate signed prints into market orders (event 1 = buy, 2 = sell)."""
    t = trades.dropna(subset=["trade_sign"]).sort_values(["ric", "ts"], kind="mergesort")
    tn = _epoch_ns(t["ts"])                             # ns since epoch
    ric = t["ric"].to_numpy()
    sgn = t["trade_sign"].to_numpy()
    new = np.ones(len(t), dtype=bool)
    new[1:] = (ric[1:] != ric[:-1]) | (sgn[1:] != sgn[:-1]) | (np.diff(tn) > gap_ms * 1e6)
    oid = np.cumsum(new)
    g = t.assign(oid=oid).groupby("oid")
    mo = pd.DataFrame({"ric": g["ric"].first(), "ts": g["ts"].first(),
                       "sign": g["trade_sign"].first(), "volume": g["volume"].sum(),
                       "prints": g.size()}).reset_index(drop=True)
    mo["event"] = np.where(mo["sign"] > 0, 1, 2)
    return mo


def _book_dict(prices, sizes):
    return {p: s for p, s in zip(prices, sizes) if p > 0 and s > 0}


def book_events(book, prints, match_window_ms=500):
    """Limit-order and cancellation events (types 3-10) for ONE security and
    ONE day.

    book   : DataFrame with ts and L1..L10 bid/ask prices and sizes, sorted by
             time, continuous phase only.
    prints : DataFrame of that day's continuous on-book trade prints with ts,
             price, volume, trade_sign (+1 buy, -1 sell).
    """
    bp = book[[f"L{k}-BidPrice" for k in range(1, N_LEVELS + 1)]].to_numpy()
    bs = book[[f"L{k}-BidSize" for k in range(1, N_LEVELS + 1)]].to_numpy()
    ap = book[[f"L{k}-AskPrice" for k in range(1, N_LEVELS + 1)]].to_numpy()
    asz = book[[f"L{k}-AskSize" for k in range(1, N_LEVELS + 1)]].to_numpy()
    tms = _epoch_ns(book["ts"]) // 1_000_000                            # ms
    ts = book["ts"].to_numpy()

    # trade prints that can consume depth: sells consume bids, buys consume asks
    pr = prints.sort_values("ts", kind="mergesort")
    p_ms = _epoch_ns(pr["ts"]) // 1_000_000
    p_side = np.where(pr["trade_sign"].to_numpy() > 0, "ask", "bid")
    p_price = pr["price"].to_numpy()
    p_left = pr["volume"].to_numpy().astype(float).copy()
    lo = 0                                       # first print still inside the window

    out_t, out_e, out_q = [], [], []
    prev_b = _book_dict(bp[0], bs[0])
    prev_a = _book_dict(ap[0], asz[0])
    prev_full_b = (bp[0] > 0).sum() == N_LEVELS
    prev_full_a = (ap[0] > 0).sum() == N_LEVELS

    for i in range(1, len(book)):
        new_b = _book_dict(bp[i], bs[i])
        new_a = _book_dict(ap[i], asz[i])
        full_b = (bp[i] > 0).sum() == N_LEVELS
        full_a = (ap[i] > 0).sum() == N_LEVELS
        t_i = tms[i]
        while lo < len(p_ms) and p_ms[lo] < t_i - match_window_ms:
            lo += 1

        for side, prev, new, pfull, nfull in (("bid", prev_b, new_b, prev_full_b, full_b),
                                              ("ask", prev_a, new_a, prev_full_a, full_a)):
            if prev == new:
                continue
            # prices visible in both snapshots
            if side == "bid":
                bound = max(min(prev) if (pfull and prev) else -np.inf,
                            min(new) if (nfull and new) else -np.inf)
                best = max(prev) if prev else None
                visible = lambda p: p >= bound
                at_or_better = lambda p: best is None or p >= best
            else:
                bound = min(max(prev) if (pfull and prev) else np.inf,
                            max(new) if (nfull and new) else np.inf)
                best = min(prev) if prev else None
                visible = lambda p: p <= bound
                at_or_better = lambda p: best is None or p <= best
            off = 0 if side == "bid" else 1      # event-code offset (bid even/ask odd)
            for p in set(prev) | set(new):
                if not visible(p):
                    continue
                d = new.get(p, 0) - prev.get(p, 0)
                if d > 0:
                    out_t.append(ts[i]); out_q.append(d)
                    out_e.append((3 if at_or_better(p) else 5) + off)
                elif d < 0:
                    removed = -d
                    # match against trade prints on this side at this price
                    j = lo
                    while removed > 0 and j < len(p_ms) and p_ms[j] <= t_i + match_window_ms:
                        if p_side[j] == side and p_price[j] == p and p_left[j] > 0:
                            take = min(removed, p_left[j])
                            p_left[j] -= take
                            removed -= take
                        j += 1
                    if removed > 0:
                        out_t.append(ts[i]); out_q.append(removed)
                        out_e.append((7 if p == best else 9) + off)
        prev_b, prev_a, prev_full_b, prev_full_a = new_b, new_a, full_b, full_a

    ev = pd.DataFrame({"ts": out_t, "event": out_e, "quantity": out_q})
    ev.attrs["unmatched_print_volume"] = float(p_left.sum())
    ev.attrs["print_volume"] = float(pr["volume"].sum())
    return ev


def identify_events(trades, book_loader, rics, days, match_window_ms=500, gap_ms=10.0):
    """All ten event types for each security and day.

    trades      : continuous on-book prints with ric, ts, date, price, volume, trade_sign
    book_loader : function (ric, day) -> continuous-phase L1-L10 snapshots for that day
    Returns events (ric, date, ts, event, quantity) and a per-day diagnostics table.
    """
    mo = market_orders(trades, gap_ms)
    mo["date"] = mo["ts"].dt.tz_convert(TZ_LOCAL).dt.date
    parts, diag = [], []
    for ric in rics:
        tr = trades[trades["ric"] == ric]
        for d in days:
            bk = book_loader(ric, d)
            if bk.empty:
                continue
            ev = book_events(bk, tr[tr["date"] == d], match_window_ms)
            matched = 1 - ev.attrs["unmatched_print_volume"] / max(ev.attrs["print_volume"], 1)
            m = mo[(mo["ric"] == ric) & (mo["date"] == d)]
            ev = pd.concat([ev, pd.DataFrame({"ts": m["ts"], "event": m["event"], "quantity": m["volume"]})])
            ev["ric"], ev["date"] = ric, d
            parts.append(ev)
            diag.append({"ric": ric, "date": d, "book updates": len(bk),
                         "print volume matched to book": matched})
    events = pd.concat(parts, ignore_index=True).sort_values(["ric", "date", "ts", "event"], kind="mergesort")
    events["name"] = events["event"].map(EVENT_NAMES)
    return events.reset_index(drop=True), pd.DataFrame(diag)
