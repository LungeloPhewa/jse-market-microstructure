"""
bar_construction.py -- Part I.A: Bar-data and candlestick plots.

Builds OHLCV bars (10-min) from both the transaction-price series (TX feed)
and the mid-price series (MD feed), plus the corresponding VWAP and
micro-price "overlay" series at both 10-min and 1-min resolution, and a
dependency-free candlestick plotting routine (no mplfinance required).

TX-based and MD-based bars are computed independently (see data_io.py) and
only ever placed on the same axes for visual overlay -- never merged as
dataframes.
"""

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Must match data_io.TZ_LOCAL. Duplicated (rather than imported) so this
# module stays independently usable on any ts-indexed DataFrame, not
# coupled to data_io's loaders specifically.
TZ_LOCAL = "Africa/Johannesburg"  # SAST = UTC+2, no daylight saving


def _localise(df, ts_col):
    """
    `ts` is stored as UTC throughout (see data_io.py), but JSE session hours
    and every reader's intuition for "the 10:00 bar" are in SAST local time.
    Since SAST is a whole-hour offset from UTC (no DST), converting to local
    time before resampling does NOT change which trades/quotes fall in which
    bar -- bar boundaries land on the same instants either way -- it only
    changes how the resulting index is labelled/plotted. Without this, bars
    and candlestick x-axes were showing UTC clock times while everything
    else (session filtering, the write-up's discussion of trading hours)
    assumes SAST, which made a correctly-filtered 09:10-15:50 SAST session
    look like it was cut off at 07:10-13:50 on the chart.

    Note: localising the index alone is NOT enough for the chart. Matplotlib
    converts datetimes to UTC day-numbers and DateFormatter formats them in
    rcParams["timezone"] (UTC by default), so plot_candles must also pass
    tz=TZ_LOCAL to its DateFormatter. That second half was missing in the
    first render, which is why the candle axes still read 07:00-14:00.
    """
    out = df.set_index(ts_col)
    out.index = out.index.tz_convert(TZ_LOCAL)
    return out


def make_ohlcv(df, price_col, vol_col, freq, ts_col="ts"):
    """Generic OHLC(V) bar builder for a single (ric, date) slice of
    either the trade-price series or the mid-price series. Bars are
    indexed in local SAST time (see _localise)."""
    s = _localise(df, ts_col)
    bars = pd.DataFrame({
        "open": s[price_col].resample(freq).first(),
        "high": s[price_col].resample(freq).max(),
        "low": s[price_col].resample(freq).min(),
        "close": s[price_col].resample(freq).last(),
    })
    if vol_col is not None:
        bars["volume"] = s[vol_col].resample(freq).sum()
    return bars.dropna(subset=["open"])


def clean_quotes(md):
    """Keep only quote updates with a valid touch: bid > 0, ask > bid and,
    where present, a micro-price inside [bid, ask]. Removes the zero /
    crossed / one-sided states that appear around the open and after the
    last trade, which otherwise produce spurious mid- and micro-price
    spikes (e.g. the GRTJ.J micro-price dropping below the last candle)."""
    if not {"bid", "ask"}.issubset(md.columns):
        return md
    ok = md["bid"].gt(0) & md["ask"].gt(md["bid"])
    if "micro" in md.columns:
        ok &= md["micro"].between(md["bid"], md["ask"])
    return md[ok]


def make_vwap_bars(trades, freq, ts_col="ts"):
    """One VWAP value per bar: sum(price*volume)/sum(volume) of all trades
    falling within the bar. This is the TX-feed "VWAP bar-data" series.
    Indexed in local SAST time (see _localise)."""
    s = _localise(trades, ts_col)
    pv = (s["price"] * s["volume"]).resample(freq).sum()
    v = s["volume"].resample(freq).sum()
    return (pv / v).rename("vwap").dropna()


def make_microprice_bars(md, freq, ts_col="ts"):
    """One micro-price value per bar: duration-weighted average of the
    micro-price over all quote updates observed within the bar. This is
    the MD-feed "micro-price bar-data" series. Indexed in local SAST time
    (see _localise)."""
    s = _localise(clean_quotes(md), ts_col).sort_index()

    def _twap(x):
        if len(x) == 0:
            return np.nan
        if len(x) == 1:
            return x.iloc[0]
        # .copy() is required here: .values can return a read-only array
        # (seen with some pandas/numpy version combinations for
        # datetime-derived computations like this one), and the next few
        # lines mutate dt[-1] in place.
        dt = x.index.to_series().diff().shift(-1).dt.total_seconds().fillna(0).values.copy()
        if dt[:-1].size and dt[:-1].mean() > 0:
            dt[-1] = dt[:-1].mean()
        else:
            dt[-1] = 1.0
        return np.average(x.values, weights=np.where(dt <= 0, 1e-9, dt))

    return s["micro"].resample(freq).apply(_twap).rename("microprice").dropna()


def plot_candles(ax, ohlc, overlay=None, overlay_label=None, title="", freq=None):
    """Minimal dependency-free candlestick chart.

    Candle width is set from the bar frequency (`freq`), not from the gap
    between the first two bars: with sparse 1-minute data the first gap can
    be several minutes, which drew every candle several minutes wide (the
    GRTJ.J midday panel in the first render). Time labels are SAST.
    """
    if freq is not None:
        bar_days = pd.Timedelta(freq).total_seconds() / 86400
    elif len(ohlc) > 1:
        bar_days = ohlc.index.to_series().diff().min().total_seconds() / 86400
    else:
        bar_days = 60 / 86400
    width = bar_days * 0.7

    if ohlc.empty:
        ax.text(0.5, 0.5, "No trades in this window", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
    for t, row in ohlc.iterrows():
        colour = "tab:green" if row["close"] >= row["open"] else "tab:red"
        # date2num of a tz-aware timestamp is the correct UTC instant; the
        # formatter below renders it in SAST. Candles are centred mid-bar.
        t_num = mdates.date2num(t) + bar_days / 2
        ax.add_line(Line2D([t_num, t_num], [row["low"], row["high"]],
                           color=colour, linewidth=0.8, zorder=1))
        lower = min(row["open"], row["close"])
        height = abs(row["close"] - row["open"])
        if height == 0:   # doji: draw a visible horizontal tick instead of a 0-height box
            ax.add_line(Line2D([t_num - width / 2, t_num + width / 2], [lower, lower],
                               color=colour, linewidth=1.2, zorder=2))
        else:
            ax.add_patch(Rectangle((t_num - width / 2, lower), width, height,
                                   facecolor=colour, edgecolor=colour, zorder=2))
    if overlay is not None and len(overlay):
        x_ov = mdates.date2num(overlay.index.to_pydatetime()) + bar_days / 2
        ax.plot(x_ov, overlay.values, color="black", linewidth=1.1,
                label=overlay_label, zorder=3)
        ax.legend(loc="upper left")
    if not ohlc.empty:
        ax.autoscale_view()
    ax.xaxis_date(tz=TZ_LOCAL)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ_LOCAL))
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_title(title)
    ax.set_xlabel("Time (SAST)")
    ax.set_ylabel("Price (ZAc)")


def _filter_window(df, start_time, end_time, ts_col="ts"):
    """Restrict a (ric, day)-filtered frame to a LOCAL (SAST) time-of-day
    window [start_time, end_time) -- e.g. start_time="09:10", end_time=
    "10:10" -- for zooming into part of the session at finer resolution.
    Filtering is done on local time, consistent with session filtering and
    every other time-of-day operation in this pipeline (see TZ_LOCAL
    above / data_io.TZ_LOCAL)."""
    local_time = df[ts_col].dt.tz_convert(TZ_LOCAL).dt.time
    start = pd.to_datetime(start_time).time()
    end = pd.to_datetime(end_time).time()
    return df[(local_time >= start) & (local_time < end)]


def plot_tx_candles_with_vwap_window(trades, ric, day, ax, start_time, end_time, freq="1min"):
    """Same as plot_tx_candles_with_vwap, but zoomed into a local-SAST
    time-of-day window (e.g. a single hour) at finer resolution -- useful
    for seeing individual bullish/bearish candles that a full 10-min,
    full-day view compresses away."""
    tx_day = trades[(trades["ric"] == ric) & (trades["date"] == day)]
    tx_window = _filter_window(tx_day, start_time, end_time)
    ohlcv = make_ohlcv(tx_window, "price", "volume", freq)
    vwap = make_vwap_bars(tx_window, freq)
    plot_candles(ax, ohlcv, overlay=vwap, overlay_label=f"VWAP ({freq})", freq=freq,
                 title=f"{ric} {day} {start_time}-{end_time}: transaction-price candles + VWAP")
    return ohlcv, vwap


def plot_mid_candles_with_microprice_window(md, ric, day, ax, start_time, end_time, freq="1min"):
    """Windowed counterpart of plot_mid_candles_with_microprice (see
    plot_tx_candles_with_vwap_window for the windowing logic)."""
    md_day = md[(md["ric"] == ric) & (md["date"] == day)]
    md_window = _filter_window(clean_quotes(md_day), start_time, end_time)
    ohlcv = make_ohlcv(md_window, "mid", None, freq)
    micro = make_microprice_bars(md_window, freq)
    plot_candles(ax, ohlcv, overlay=micro, overlay_label=f"Micro-price ({freq})", freq=freq,
                 title=f"{ric} {day} {start_time}-{end_time}: mid-price candles + micro-price")
    return ohlcv, micro


def plot_tx_candles_with_vwap(trades, ric, day, ax, freq="10min"):
    """Convenience wrapper: transaction-price OHLCV candles for one
    (ric, day), with the matching-frequency VWAP line overlaid."""
    tx_day = trades[(trades["ric"] == ric) & (trades["date"] == day)]
    ohlcv = make_ohlcv(tx_day, "price", "volume", freq)
    vwap = make_vwap_bars(tx_day, freq)
    plot_candles(ax, ohlcv, overlay=vwap, overlay_label=f"VWAP ({freq})", freq=freq,
                 title=f"{ric} {day}: transaction-price candles + VWAP")
    return ohlcv, vwap


def plot_mid_candles_with_microprice(md, ric, day, ax, freq="10min"):
    """Convenience wrapper: mid-price OHLCV candles for one (ric, day),
    with the matching-frequency micro-price line overlaid."""
    md_day = clean_quotes(md[(md["ric"] == ric) & (md["date"] == day)])
    ohlcv = make_ohlcv(md_day, "mid", None, freq)
    micro = make_microprice_bars(md_day, freq)
    plot_candles(ax, ohlcv, overlay=micro, overlay_label=f"Micro-price ({freq})", freq=freq,
                 title=f"{ric} {day}: mid-price candles + micro-price")
    return ohlcv, micro
