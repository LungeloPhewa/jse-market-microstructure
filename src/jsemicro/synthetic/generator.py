"""Synthetic tick data in the Refinitiv (TRTH) raw-export format.

The real data behind this project are licensed Refinitiv exports and cannot be
published. This module writes two files with the *same layout* as the raw
exports, so that every stage of the pipeline -- from flattening the FID rows
to the Hawkes calibration -- can be run, tested and demonstrated by anyone:

``JSE-AGL-GRT-Transactions-Raw-SYNTHETIC.csv.gz``
    one header row per message (``Type = Raw-TAS``) followed by one
    ``Type = FID`` row per field, exactly like the real Transactions export:
    trade prints (``TRDPRC_1``, ``TRDVOL_1``, ``ACVOL_1``), top-of-book quote
    updates (``BID``/``BIDSIZE``, ``ASK``/``ASKSIZE``), market-state markers
    (``PRC_QL2``/``PRC_QL3``), auction uncross prints (``ACT_FLAG1 = U``),
    off-book trades (``OFFBK_VOL``), index records and sync messages.

``JSE-AGL-GRT-MarketDepth-SYNTHETIC.csv.gz``
    one row per change of the ten-level book (``L1``-``L10`` bid/ask price and
    size), already wide, like the real Market Depth export.

The market
----------
For each security and trading day a limit order book on an integer tick grid
(prices in ZAc) is driven by a 10-variate Hawkes process
(:mod:`jsemicro.synthetic.hawkes_sim`) over the ten event types of
Jericevich, Chang & Gebbie (2021): buy/sell market orders, aggressive and
passive limit orders and cancellations on each side. Market orders walk the
book and print once per price level they consume (same millisecond), so
the Assignment 3 rule "same-sign prints within 10 ms are one market order"
recovers them. A small share of market orders are blocks larger than the
best level, which produce the depth shocks studied in Part II.

Times follow the JSE trading calendar in SAST (written to the file in GMT, as
the real ``Date[G]``/``Time[G]`` fields are): pre-open 08:00, opening auction
08:30, continuous trading 09:00-16:50, closing auction 16:50-17:00, post-close
off-book reporting, and on one day an intraday auction 12:00-12:15.

Everything is deterministic given ``SEED``. The point is not to reproduce the
JSE's statistics exactly but to exercise every code path with data that look
and behave like the real thing (heavy-tailed volumes, clustered arrivals,
mean-reverting spreads, a U-shaped intraday pattern).
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import hawkes_sim

SEED = 20130902
N_LEVELS = 10
GMT_OFFSET = 2                         # SAST = GMT + 2
DAYS = [dt.date(2013, 9, d) for d in (2, 3, 4, 5, 6)]
INTRADAY_AUCTION_DAY = dt.date(2013, 9, 5)
TX_NAME = "JSE-AGL-GRT-Transactions-Raw-SYNTHETIC.csv.gz"
MD_NAME = "JSE-AGL-GRT-MarketDepth-SYNTHETIC.csv.gz"

# Event types (0-based) -- same order as jsemicro.hawkes.order_events
BUY_MO, SELL_MO, AGG_BID_LO, AGG_ASK_LO, PAS_BID_LO, PAS_ASK_LO, \
    AGG_BID_CX, AGG_ASK_CX, PAS_BID_CX, PAS_ASK_CX = range(10)

TX_HEADER = ["#RIC", "Date[G]", "Time[G]", "GMT Offset", "Type", "MSG/FID Number",
             "Message Type", "FID Name", "FID Value", "FID Enum String", "PE Code",
             "Sequence Number", "Source RIC", "Number of FIDs"]
MD_HEADER = ["#RIC", "Date[G]", "Time[G]", "GMT Offset", "Type"] + [
    f"L{k}-{f}" for k in range(1, N_LEVELS + 1) for f in ("BidPrice", "BidSize", "AskPrice", "AskSize")]


@dataclass
class Security:
    """Parameters of one synthetic security."""
    ric: str
    price0: int                 # opening price, ZAc
    spread0: int                # typical spread in ticks
    depth_median: float         # median resting size per level (shares)
    mo_median: float            # median market-order size (shares)
    rate: float                 # mean events per second (baseline, before excitation)
    daily_vol: float = 0.015    # daily volatility of the fair value the quotes revert to
    seed: int = 0
    share: dict = field(default_factory=dict)


SECURITIES = [
    Security("AGLJ.J", price0=24_500, spread0=12, depth_median=300, mo_median=120, rate=0.9, seed=1),
    Security("GRTJ.J", price0=2_420, spread0=2, depth_median=6_000, mo_median=1_500, rate=0.18, seed=2),
]

# Baseline mix of the ten event types (sums to one)
EVENT_SHARE = np.array([0.07, 0.07, 0.09, 0.09, 0.15, 0.15, 0.07, 0.07, 0.12, 0.12])


def excitation():
    """Branching ratios Gamma and decay rates beta of the order-flow model.

    Diagonal self-excitation (order splitting, clustering), market orders
    exciting same-side market orders and the refill of the side they consumed,
    and aggressive limit orders exciting cancellations on the same side
    (fleeting liquidity). Spectral radius about 0.55.
    """
    G = np.zeros((10, 10))
    np.fill_diagonal(G, 0.25)
    G[BUY_MO, BUY_MO] = G[SELL_MO, SELL_MO] = 0.30
    G[AGG_ASK_LO, BUY_MO] = G[AGG_BID_LO, SELL_MO] = 0.30          # refill after a market order
    G[AGG_BID_LO, BUY_MO] = G[AGG_ASK_LO, SELL_MO] = 0.10          # follow-through
    G[AGG_BID_CX, AGG_BID_LO] = G[AGG_ASK_CX, AGG_ASK_LO] = 0.15   # fleeting liquidity
    G[PAS_BID_CX, PAS_BID_LO] = G[PAS_ASK_CX, PAS_ASK_LO] = 0.10
    beta = np.full((10, 10), 1.0)
    beta[:2, :2] = 3.0                  # market orders cluster on sub-second scales
    beta[2:4, :2] = 5.0                 # refills are fast
    beta[4:, 4:] = 0.2                  # passive activity is slow
    return G * beta, beta


# ----------------------------------------------------------------------------
# Order book
# ----------------------------------------------------------------------------

class Book:
    """Price-level limit order book (aggregate size per price)."""

    def __init__(self, rng, sec: Security):
        self.rng, self.sec = rng, sec
        self.anchor = float(sec.price0)      # slowly moving "fair value" that quotes revert to
        mid = sec.price0
        half = max(sec.spread0 // 2, 1)
        self.bids = {mid - half - i: self._size() for i in range(N_LEVELS + 2)}
        self.asks = {mid + half + i: self._size() for i in range(N_LEVELS + 2)}

    def _size(self, scale=1.0):
        return int(max(1, round(self.rng.lognormal(np.log(self.sec.depth_median * scale), 0.8))))

    def best_bid(self): return max(self.bids)
    def best_ask(self): return min(self.asks)

    def levels(self, side):
        book = self.bids if side == "bid" else self.asks
        prices = sorted(book, reverse=(side == "bid"))[:N_LEVELS]
        return [(p, book[p]) for p in prices]

    def snapshot(self):
        b, a = self.levels("bid"), self.levels("ask")
        b += [(0, 0)] * (N_LEVELS - len(b))
        a += [(0, 0)] * (N_LEVELS - len(a))
        return tuple(x for k in range(N_LEVELS) for x in (b[k][0], b[k][1], a[k][0], a[k][1]))

    def _refill(self):
        """Background liquidity provision: keep at least twelve levels on each
        side and at least six of them within fifteen ticks of the best quote,
        so that the book does not hollow out over a long simulated day."""
        for side, book in (("bid", self.bids), ("ask", self.asks)):
            sgn = -1 if side == "bid" else 1
            best = max(book) if side == "bid" else min(book)
            near = sum(1 for p in book if 0 <= sgn * (p - best) <= 15)
            while near < 6:
                p = best + sgn * int(self.rng.integers(1, 16))
                if p not in book:
                    near += 1
                book[p] = book.get(p, 0) + self._size()
            while len(book) < N_LEVELS + 2:
                edge = min(book) if side == "bid" else max(book)
                p = edge + sgn * (1 + int(self.rng.integers(0, 2)))
                book[p] = book.get(p, 0) + self._size(1.5)

    def market_order(self, side, volume):
        """Execute a market order; returns the list of (price, volume) prints."""
        book = self.asks if side == "buy" else self.bids
        prints = []
        while volume > 0 and len(book) > 1:
            p = min(book) if side == "buy" else max(book)
            take = min(volume, book[p])
            prints.append((p, take))
            book[p] -= take
            volume -= take
            if book[p] == 0:
                del book[p]
        self._refill()
        self._uncross()
        return prints

    def _uncross(self):
        """Refills can never cross the book (they go behind the best quote),
        but guard against it anyway."""
        while self.best_bid() >= self.best_ask():
            del self.bids[self.best_bid()]

    def limit_order(self, side, aggressive):
        """Add a limit order. Aggressive orders join or improve the best quote;
        when the spread is wider than usual they land inside it (liquidity
        providers compete the spread back down), passive orders go 1-10 ticks
        behind the best quote."""
        bb, ba = self.best_bid(), self.best_ask()
        spread = ba - bb
        # Quotes lean against deviations from fair value: if the mid is above
        # the anchor, sellers improve more readily and buyers only join.
        lean = ((bb + ba) / 2 - self.anchor) / max(self.sec.spread0, 1)
        lean = lean if side == "ask" else -lean
        if aggressive:
            improve_p = min(0.95, max(0.05, 0.4 + 0.25 * lean))
            if spread > 1 and (spread > self.sec.spread0 or self.rng.random() < improve_p):
                gap = int(self.rng.integers(1, spread))                # strictly inside the spread
                if spread <= self.sec.spread0:
                    gap = 1
                elif lean < 1:
                    gap = min(gap, max(1, spread // 2))
            else:
                gap = 0                                                # join the best quote
            p = bb + gap if side == "bid" else ba - gap
        else:
            k = min(int(self.rng.geometric(0.35)), N_LEVELS)
            p = bb - k if side == "bid" else ba + k
        book = self.bids if side == "bid" else self.asks
        book[p] = book.get(p, 0) + self._size()

    def cancel(self, side, aggressive):
        book = self.bids if side == "bid" else self.asks
        lv = self.levels(side)
        if len(lv) < 3:
            return
        if aggressive:
            p, s = lv[0]
            if (self.best_ask() - self.best_bid()) > 2 * self.sec.spread0:
                return                     # nobody pulls the best quote when the spread is already wide
        else:
            p, s = lv[min(1 + int(self.rng.geometric(0.4)) - 1, len(lv) - 1)]
        q = s if self.rng.random() < 0.6 else max(1, int(s * self.rng.uniform(0.2, 0.9)))
        book[p] -= q
        if book[p] <= 0:
            del book[p]
        self._refill()


# ----------------------------------------------------------------------------
# Session simulation
# ----------------------------------------------------------------------------

def _sast(day, hh, mm, ss=0.0):
    return dt.datetime.combine(day, dt.time(hh, mm)) + dt.timedelta(seconds=ss)


def simulate_day(sec: Security, day: dt.date, seed: int):
    """Simulate one security-day.

    Returns a dict with lists of transaction messages and book snapshots, all
    stamped in SAST (naive datetimes); conversion to GMT happens on writing.
    """
    rng = np.random.default_rng(seed)
    book = Book(rng, sec)
    tx, md = [], []                       # (time, kind, fields) and (time, snapshot)
    acvol = 0
    G_alpha, beta = excitation()
    open_t = _sast(day, 9, 0, rng.uniform(0.5, 8))
    close_t = _sast(day, 16, 50)
    gaps = []
    if day == INTRADAY_AUCTION_DAY:
        gaps = [(_sast(day, 12, 0), _sast(day, 12, 15, rng.uniform(0.5, 5)))]

    # --- markers, pre-open and opening auction -------------------------------
    for t, q2, q3 in ((_sast(day, 8, 0, 0.3), 102, 0), (_sast(day, 8, 30, 0.3), 238, 0)):
        tx.append((t, "marker", {"PRC_QL2": q2, "PRC_QL3": q3}))
    for k in range(5):                    # the auction book builds up (not crossed)
        md.append((_sast(day, 8, 30, 60 + 300 * k + rng.uniform(0, 60)), book.snapshot()))
    uncross_vol = int(sec.mo_median * rng.uniform(40, 80))
    open_price = (book.best_bid() + book.best_ask()) // 2
    acvol += uncross_vol
    tx.append((open_t, "marker", {"PRC_QL2": 105, "PRC_QL3": 0}))
    tx.append((open_t, "trade", {"TRDPRC_1": open_price, "TRDVOL_1": uncross_vol, "ACVOL_1": acvol,
                                 "ACT_FLAG1": "U", "AUCTIONVOL": uncross_vol, "PRCTCK_1": 1}))
    tx.append((open_t, "quote", _l1_fields(book)))
    md.append((open_t + dt.timedelta(milliseconds=2), book.snapshot()))

    # --- continuous trading: Hawkes-driven order flow ----------------------------
    T = (close_t - open_t).total_seconds()
    mu = sec.rate * EVENT_SHARE
    times, types = hawkes_sim.simulate(mu, G_alpha, beta, T, seed=seed)
    last_l1 = _l1(book)
    last_trade = open_price
    vol_per_event = sec.daily_vol * sec.price0 / np.sqrt(max(len(times), 1))
    for s, k in zip(times, types):
        book.anchor += rng.normal(0.0, vol_per_event)
        t = open_t + dt.timedelta(seconds=float(s))
        t = t.replace(microsecond=(t.microsecond // 1000) * 1000)
        if any(g0 <= t < g1 for g0, g1 in gaps):
            continue
        if k in (BUY_MO, SELL_MO):
            side = "buy" if k == BUY_MO else "sell"
            best = book.best_ask() if side == "buy" else book.best_bid()
            depth1 = (book.asks if side == "buy" else book.bids)[best]
            if rng.random() < 0.03:                     # block order: walks the book
                vol = int(depth1 * rng.uniform(1.2, 3.0))
            else:
                vol = int(max(1, round(rng.lognormal(np.log(sec.mo_median), 1.0))))
                vol = min(vol, depth1) if rng.random() < 0.85 else vol
            for p, v in book.market_order(side, vol):
                acvol += v
                tick = 1 if p > last_trade else 2 if p < last_trade else 0
                tx.append((t, "trade", {"TRDPRC_1": p, "TRDVOL_1": v, "ACVOL_1": acvol, "PRCTCK_1": tick}))
                last_trade = p
        elif k in (AGG_BID_LO, AGG_ASK_LO, PAS_BID_LO, PAS_ASK_LO):
            book.limit_order("bid" if k in (AGG_BID_LO, PAS_BID_LO) else "ask", k in (AGG_BID_LO, AGG_ASK_LO))
        else:
            book.cancel("bid" if k in (AGG_BID_CX, PAS_BID_CX) else "ask", k in (AGG_BID_CX, AGG_ASK_CX))
        # Market Depth row for every book change; the depth feed is stamped a
        # few milliseconds after the trade feed, as in the real exports.
        md.append((t + dt.timedelta(milliseconds=int(rng.integers(1, 6))), book.snapshot()))
        l1 = _l1(book)
        if l1 != last_l1:
            tx.append((t, "quote", _l1_fields(book, last_l1)))
            last_l1 = l1
        if rng.random() < 0.0005:
            tx.append((t, "sync", {}))

    # --- intraday auction markers ---------------------------------------------------
    for g0, g1 in gaps:
        tx.append((g0, "marker", {"PRC_QL2": 238, "PRC_QL3": 138}))
        p = (book.best_bid() + book.best_ask()) // 2
        v = int(sec.mo_median * rng.uniform(5, 15))
        acvol += v
        tx.append((g1, "marker", {"PRC_QL2": 105, "PRC_QL3": 0}))
        tx.append((g1, "trade", {"TRDPRC_1": p, "TRDVOL_1": v, "ACVOL_1": acvol, "ACT_FLAG1": "U",
                                 "AUCTIONVOL": v, "PRCTCK_1": 0}))

    # --- closing auction, close and off-book reporting ------------------------------
    tx.append((close_t, "marker", {"PRC_QL2": 238, "PRC_QL3": 291}))
    for k in range(3):
        md.append((close_t + dt.timedelta(seconds=120 * k + rng.uniform(1, 60)), book.snapshot()))
    end_t = _sast(day, 17, 0, rng.uniform(0.5, 9))
    close_price = (book.best_bid() + book.best_ask()) // 2
    v = int(sec.mo_median * rng.uniform(60, 150))
    acvol += v
    tx.append((end_t, "marker", {"PRC_QL2": 94, "PRC_QL3": 131}))
    tx.append((end_t, "trade", {"TRDPRC_1": close_price, "TRDVOL_1": v, "ACVOL_1": acvol,
                                "ACT_FLAG1": "U", "AUCTIONVOL": v, "PRCTCK_1": 0}))
    md.append((end_t + dt.timedelta(milliseconds=3), book.snapshot()))
    offbk = 0
    for _ in range(int(rng.integers(1, 4))):
        v = int(sec.mo_median * rng.uniform(100, 400))
        offbk += v
        t = _sast(day, 17, 20, rng.uniform(0, 3000))
        tx.append((t, "trade", {"TRDPRC_1": close_price + int(rng.integers(-3, 4)), "TRDVOL_1": v,
                                "OFFBK_VOL": offbk, "PRCTCK_1": 0}))
    tx.append((_sast(day, 18, 15, 0.4), "marker", {"PRC_QL2": 94, "PRC_QL3": 320}))
    tx.append((_sast(day, 18, 30, 0.4), "marker", {"PRC_QL2": 94, "PRC_QL3": 0}))
    tx.sort(key=lambda r: r[0])
    md.sort(key=lambda r: r[0])
    return {"tx": tx, "md": md}


def _l1(book):
    bb, ba = book.best_bid(), book.best_ask()
    return bb, book.bids[bb], ba, book.asks[ba]


def _l1_fields(book, previous=None):
    """FIDs of a top-of-book quote update: only the side(s) that changed."""
    bb, bs, ba, asz = _l1(book)
    out = {"MID_PRICE": (bb + ba) / 2}
    if previous is None or (bb, bs) != previous[:2]:
        out.update(BID=bb, BIDSIZE=bs)
    if previous is None or (ba, asz) != previous[2:]:
        out.update(ASK=ba, ASKSIZE=asz)
    return out


# ----------------------------------------------------------------------------
# Writing the raw exports
# ----------------------------------------------------------------------------

def _gzip_text(path: Path):
    """Text handle to a gzip file with a fixed header (mtime 0, no file name),
    so the same data always give the same bytes and the same checksum."""
    raw = open(path, "wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6)
    return io.TextIOWrapper(gz, newline="", encoding="utf-8"), raw


def _gmt_date_time(t_sast: dt.datetime):
    g = t_sast - dt.timedelta(hours=GMT_OFFSET)
    return g.strftime("%d-%b-%Y").upper(), g.strftime("%H:%M:%S.") + f"{g.microsecond // 1000:03d}"


def _write_tx(path: Path, streams):
    seq = {}
    fh, raw = _gzip_text(path)
    with raw, fh:
        w = csv.writer(fh)
        w.writerow(TX_HEADER)
        for ric, t, kind, fields in streams:
            date_s, time_s = _gmt_date_time(t)
            seq[ric] = seq.get(ric, 0) + int(1 + (hashlib.md5(f"{ric}{t}".encode()).digest()[0] % 3))
            msg = "VERIFY_SYNC_PC" if kind == "sync" else "UPDATE_PC"
            fields = dict(fields)
            if kind != "sync":
                fields["SEQNUM"] = seq[ric]
            w.writerow([ric, date_s, time_s, f"+{GMT_OFFSET}", "Raw-TAS", "", msg, "", "", "", "", "",
                        "", len(fields)])
            for name, value in fields.items():
                w.writerow(["", "", "", "", "FID", "", "", name, value, "", "", "", "", ""])


def _write_md(path: Path, rows):
    fh, raw = _gzip_text(path)
    with raw, fh:
        w = csv.writer(fh)
        w.writerow(MD_HEADER)
        for ric, t, snap in rows:
            date_s, time_s = _gmt_date_time(t)
            w.writerow([ric, date_s, time_s, f"+{GMT_OFFSET}", "Market Depth", *snap])


def write_raw_exports(raw_dir: str | Path, days=None, securities=None) -> tuple[Path, Path]:
    """Simulate all securities and days and write the two raw exports.

    Returns the paths of the Transactions and Market Depth files.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    days = days or DAYS
    securities = securities or SECURITIES
    tx_rows, md_rows = [], []
    rng = np.random.default_rng(SEED)
    for day in days:
        # an index record or two, which the pipeline must filter out
        for hh in (7, 12):
            tx_rows.append((".JTOPI", _sast(day, hh, 0, 0.9), "index", {"TRDPRC_1": round(40000.0 + rng.normal(0, 50), 2)}))
        for sec in securities:
            seed = SEED + 1000 * sec.seed + day.toordinal() % 1000
            sim = simulate_day(sec, day, seed)
            tx_rows += [(sec.ric, t, kind, f) for t, kind, f in sim["tx"]]
            md_rows += [(sec.ric, t, snap) for t, snap in sim["md"]]
    # the real exports are ordered by RIC, then time
    order = {".JTOPI": 0, "AGLJ.J": 1, "GRTJ.J": 2}
    tx_rows.sort(key=lambda r: (order.get(r[0], 9), r[1]))
    md_rows.sort(key=lambda r: (order.get(r[0], 9), r[1]))
    tx_path, md_path = raw_dir / TX_NAME, raw_dir / MD_NAME
    _write_tx(tx_path, tx_rows)
    _write_md(md_path, md_rows)
    print(f"Synthetic exports: {len(tx_rows):,} transaction messages, {len(md_rows):,} depth rows -> {raw_dir}")
    return tx_path, md_path
