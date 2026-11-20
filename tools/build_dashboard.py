"""Build the results dashboard (dashboard/index.html + dashboard/assets/*.svg).

The dashboard is a single static page published at <site>/dashboard/. It is
built from dashboard/data/results.json, a small file of aggregate results
(no market data) written from the real-data run, so CI can publish it without
the licensed data. Rebuild after a new real-data run with:

    python tools/build_dashboard.py

The same chart functions (with fmt="png") produce the LinkedIn carousel images.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"
COL = {"AGLJ.J": "#2457a7", "GRTJ.J": "#e08a1e"}
NAME = {"AGLJ.J": "Anglo American", "GRTJ.J": "Growthpoint"}
INK, MUTED, GRID = "#1b2a4a", "#6b7280", "#e5e7eb"


def style(scale=1.0):
    plt.rcParams.update({
        "font.family": "Inter", "font.size": 10 * scale, "axes.titlesize": 11 * scale,
        "axes.labelsize": 10 * scale, "xtick.labelsize": 9 * scale, "ytick.labelsize": 9 * scale,
        "legend.fontsize": 9 * scale, "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
    })


def chart_spreads(R, ax):
    """Stacked bars: effective spread = realised spread (to liquidity providers)
    + price impact (lost to informed traders), in bps."""
    rics = ["AGLJ.J", "GRTJ.J"]
    sp = R["spread_stats"]
    real = [sp[r]["realised_bps"] for r in rics]
    imp = [sp[r]["impact_bps"] for r in rics]
    x = range(len(rics))
    ax.bar(x, imp, width=0.55, color=[COL[r] for r in rics], label="Price impact (adverse selection)")
    ax.bar(x, real, bottom=imp, width=0.55, color=[COL[r] for r in rics], alpha=0.35,
           label="Realised spread (earned by liquidity providers)")
    for i, r in enumerate(rics):
        tot = sp[r]["effective_bps"]
        ax.text(i, tot + 0.25, f"{tot:.1f} bps", ha="center", va="bottom", fontweight="bold", color=INK)
        ax.text(i, imp[i] / 2, f"{imp[i] / tot:.0%}\nimpact", ha="center", va="center", color="white",
                fontsize=plt.rcParams["font.size"] * 0.9)
    ax.set_xticks(list(x), [f"{NAME[r]}\n({r})" for r in rics])
    ax.set_ylabel("Effective spread (bps)")
    ax.set_ylim(0, max(sp[r]["effective_bps"] for r in rics) * 1.38)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")


def chart_size(R, ax):
    """Indirect cost against trade size (log-log): fitted power laws against the
    square-root law, each anchored at the smallest size bin."""
    import numpy as np
    for r in ["AGLJ.J", "GRTJ.J"]:
        f = R["sqrt_law"][r]
        v = np.logspace(np.log10(f["vol_lo"]), np.log10(f["vol_hi"]), 50)
        ax.plot(v, f["ind_lo_bps"] * (v / f["vol_lo"]) ** f["slope"], color=COL[r], linewidth=2.4,
                label=f"{NAME[r]}: exponent {f['slope']:.2f}")
        ax.plot(v, f["ind_lo_bps"] * (v / f["vol_lo"]) ** 0.5, color=COL[r], linewidth=1.2, linestyle="--",
                alpha=0.7)
    ax.plot([], [], color=MUTED, linestyle="--", label="Square-root law (exponent 0.5)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(fmt)
        a.set_minor_formatter(NullFormatter())
    ax.set_yticks([2, 3, 5, 10, 20, 40])
    ax.set_xlabel("Trade size (shares, log scale)")
    ax.set_ylabel("Price impact after 60 s (bps, log)")
    ax.legend(loc="upper left")


def chart_intraday(R, ax, what="volatility"):
    """Intraday pattern relative to the day's average (1 = average 10-minute bucket)."""
    import numpy as np
    for r in ["AGLJ.J", "GRTJ.J"]:
        c = R["seasonality"][r][what]
        x = np.arange(len(c["t"]))
        ax.plot(x, c["y"], color=COL[r], linewidth=2, label=NAME[r])
    t = R["seasonality"]["AGLJ.J"][what]["t"]
    ticks = [i for i, s in enumerate(t) if s.endswith(":00") or s == t[0]]
    ax.set_xticks(ticks, [t[i] for i in ticks])
    ax.axhline(1, color=MUTED, linestyle=":", linewidth=1)
    ax.set_ylabel("× daily average")
    ax.set_xlabel("Time of day (SAST)")
    ax.legend(loc="upper left" if what == "volume" else "upper right")


def chart_hawkes(R, ax):
    """Three key excitation effects: branching ratio, labelled with half-life."""
    e = R["hawkes_effects"]
    labels = ["Buy market order →\nask side refilled", "Sell market order →\nbest bid withdrawn",
              "Best bid withdrawn →\nbid side requoted"]
    keys = ["buy_mo_to_agg_ask_lo", "sell_mo_to_agg_bid_cancel", "agg_bid_cancel_to_agg_bid_lo"]
    y = range(len(keys))
    h = 0.36
    for j, r in enumerate(["AGLJ.J", "GRTJ.J"]):
        vals = [e[k][r] for k in keys]
        ax.barh([i + (j - 0.5) * h for i in y], vals, height=h, color=COL[r], label=NAME[r])
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Branching ratio (events triggered per source event, median day)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)


def save(fig, name, fmt="svg", dpi=200, folder=None):
    folder = folder or DASH / "assets"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return path


def build_charts(R, fmt="svg", folder=None, scale=1.0, size=(6.4, 4.0)):
    style(scale)
    out = {}
    for name, fn in (("spreads", chart_spreads), ("size", chart_size), ("hawkes", chart_hawkes)):
        fig, ax = plt.subplots(figsize=size)
        fn(R, ax)
        out[name] = save(fig, name, fmt, folder=folder)
    for what in ("volatility", "spread", "volume"):
        fig, ax = plt.subplots(figsize=size)
        chart_intraday(R, ax, what)
        out[f"intraday_{what}"] = save(fig, f"intraday_{what}", fmt, folder=folder)
    return out


BOOK = "https://lungelophewa.github.io/jse-market-microstructure"
FIGS = [
    ("06-cost-of-trading", "fig-spreads", "Effective vs quoted spreads", "The cost of trading"),
    ("06-cost-of-trading", "fig-sqrt-law", "Cost decomposition by trade size", "The cost of trading"),
    ("05-impact-seasonality-resilience", "fig-season-agl", "Intraday seasonality, AGLJ.J", "Impact, seasonality, resilience"),
    ("05-impact-seasonality-resilience", "fig-resilience", "Order-book resilience after large trades", "Impact, seasonality, resilience"),
    ("08-hawkes", "fig-bubble-agl", "Hawkes excitation bubble plots, AGLJ.J", "A Hawkes model of order flow"),
    ("03-order-flow", "fig-sign-acf", "Persistence of trade signs", "Order flow and trade arrivals"),
]


def page(R):
    sp, sq, rho, eff = R["spread_stats"], R["sqrt_law"], R["hawkes_rho_days"], R["hawkes_effects"]
    ev = R["events_per_day"]
    season = R["seasonality"]
    vmax = {r: max(season[r]["volatility"]["y"][:1]) for r in season}
    kpis = [
        ("Effective spread", f"{sp['AGLJ.J']['effective_bps']:.1f} <small>bps</small>", f"{sp['GRTJ.J']['effective_bps']:.1f} <small>bps</small>", "average cost of a round trip, relative to the mid-price"),
        ("Lost to adverse selection", f"{sp['AGLJ.J']['impact_bps'] / sp['AGLJ.J']['effective_bps']:.0%}", f"{sp['GRTJ.J']['impact_bps'] / sp['GRTJ.J']['effective_bps']:.0%}", "share of the spread that becomes a permanent price move within 60 s"),
        ("Size exponent of impact", f"{sq['AGLJ.J']['slope']:.2f}", f"{sq['GRTJ.J']['slope']:.2f}", "far below the square-root law's 0.5: large orders are split"),
        ("Volatility in the first 10 min", f"{vmax['AGLJ.J']:.1f}×", f"{vmax['GRTJ.J']:.1f}×", "relative to the average 10-minute bucket of the day"),
        ("Order-book events per day", f"{ev['AGLJ.J'] / 1000:,.0f}k", f"{ev['GRTJ.J'] / 1000:,.0f}k", "limit orders, cancellations and market orders"),
        ("Hawkes spectral radius", f"{rho['AGLJ.J'][1]:.2f}", f"{rho['GRTJ.J'][1]:.2f}", "median day; near 1 means most events are reactions to earlier events"),
        ("Refill half-life after a buy", f"{1000 * eff['half_life_buy_mo_to_agg_ask_lo']['AGLJ.J']:.0f} <small>ms</small>", f"{1000 * eff['half_life_buy_mo_to_agg_ask_lo']['GRTJ.J']:.0f} <small>ms</small>", "time for the excitation of ask-side requoting to halve"),
        ("Trade-sign autocorrelation", f"{R['sign_acf1']['AGLJ.J']:.2f}", f"{R['sign_acf1']['GRTJ.J']:.2f}", "lag 1; order flow is persistent (large orders are split)"),
    ]
    tiles = "\n".join(f"""      <div class="kpi"><div class="kpi-t">{t}</div>
        <div class="kpi-v"><span class="a">{a}</span><span class="g">{g}</span></div>
        <div class="kpi-n">{n}</div></div>""" for t, a, g, n in kpis)
    figs = "\n".join(f"""      <a class="fig" href="../chapters/{ch}.html#{fid}">
        <img loading="lazy" src="../chapters/{ch}_files/figure-html/{fid}-output-1.png" alt="{cap}">
        <span>{cap}</span><small>{part}</small></a>""" for ch, fid, cap, part in FIGS)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JSE Microstructure Results</title>
<meta name="description" content="Key results of a reproducible market-microstructure study of Anglo American and Growthpoint on the JSE.">
<style>
  :root {{ --ink:{INK}; --muted:{MUTED}; --line:{GRID}; --bg:#f5f7fb; --card:#ffffff; --a:{COL['AGLJ.J']}; --g:{COL['GRTJ.J']}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  a {{ color:var(--a); }}
  header {{ background:var(--ink); color:#fff; padding:36px 16px 30px; }}
  .wrap {{ max-width:1120px; margin:0 auto; padding:0 16px; }}
  header .wrap {{ padding:0; }}
  h1 {{ margin:0 0 6px; font-size:clamp(24px,4vw,34px); line-height:1.15; letter-spacing:-0.01em; }}
  header p {{ margin:0; color:#c9d3e6; max-width:760px; }}
  .links {{ margin-top:18px; display:flex; gap:10px; flex-wrap:wrap; }}
  .btn {{ display:inline-block; padding:8px 14px; border-radius:8px; background:#fff; color:var(--ink); text-decoration:none; font-weight:600; font-size:14px; }}
  .btn.ghost {{ background:transparent; color:#fff; border:1px solid #4b5d82; }}
  h2 {{ font-size:13px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:34px 0 12px; }}
  .legend {{ display:flex; gap:18px; font-size:13px; color:var(--muted); margin:-4px 0 12px; flex-wrap:wrap; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:12px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }}
  .kpi-t {{ font-size:13px; font-weight:600; }}
  .kpi-v {{ display:flex; gap:18px; margin:6px 0 4px; font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; }}
  .kpi-v small {{ font-size:14px; font-weight:600; }}
  .kpi-v .a {{ color:var(--a); }} .kpi-v .g {{ color:var(--g); }}
  .kpi-n {{ font-size:12.5px; color:var(--muted); line-height:1.4; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr)); gap:14px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
  .card h3 {{ margin:0 0 4px; font-size:16px; }}
  .card p {{ margin:0 0 10px; color:var(--muted); font-size:13.5px; }}
  .card img {{ width:100%; height:auto; display:block; }}
  .figs {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:12px; }}
  .fig {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px; text-decoration:none; color:var(--ink); display:flex; flex-direction:column; gap:6px; }}
  .fig img {{ width:100%; aspect-ratio:16/10; object-fit:cover; object-position:top; border-radius:6px; border:1px solid var(--line); }}
  .fig span {{ font-weight:600; font-size:14px; }} .fig small {{ color:var(--muted); }}
  .fig:hover {{ border-color:var(--a); }}
  footer {{ color:var(--muted); font-size:13px; padding:30px 16px 40px; }}
  footer p {{ max-width:860px; }}
</style>
</head>
<body>
<header><div class="wrap">
  <h1>Market Microstructure of the JSE: key results</h1>
  <p>Anglo American (AGLJ.J) and Growthpoint (GRTJ.J), {R['period']}: {R['n_trades']:,} trades and {R['n_quotes'] / 1e6:.1f} million top-of-book updates, from raw tick data to trading costs and a Hawkes model of order flow.</p>
  <div class="links">
    <a class="btn" href="../index.html">Read the full study</a>
    <a class="btn ghost" href="../chapters/09-reproducibility.html">How it is reproduced</a>
  </div>
</div></header>

<main class="wrap">
  <h2>Headline numbers</h2>
  <div class="legend"><span><span class="dot" style="background:var(--a)"></span>Anglo American (AGLJ.J)</span><span><span class="dot" style="background:var(--g)"></span>Growthpoint (GRTJ.J)</span></div>
  <div class="kpis">
{tiles}
  </div>

  <h2>Findings</h2>
  <div class="cards">
    <div class="card"><h3>Same one-cent tick, twice the cost</h3>
      <p>Growthpoint's price is a tenth of Anglo's, so its one-cent tick is ten times larger in relative terms. Its spread is about twice as wide in basis points, and a larger share of it goes to liquidity providers rather than to price impact.</p>
      <img src="assets/spreads.svg" alt="Effective spread decomposition"></div>
    <div class="card"><h3>Size matters less than expected</h3>
      <p>The price impact of a single trade grows very slowly with its size, far below the square-root law (dashed), because large orders are split into small pieces.</p>
      <img src="assets/size.svg" alt="Price impact against trade size"></div>
    <div class="card"><h3>When you trade matters: volatility</h3>
      <p>Mid-price volatility is highest just after the open and falls through the morning.</p>
      <img src="assets/intraday_volatility.svg" alt="Intraday volatility"></div>
    <div class="card"><h3>When you trade matters: spreads</h3>
      <p>Spreads are widest after the open and narrow towards the close.</p>
      <img src="assets/intraday_spread.svg" alt="Intraday spreads"></div>
    <div class="card"><h3>Order flow reacts to itself</h3>
      <p>A ten-variate Hawkes model fitted to every day finds near-critical self-excitation. Market orders are followed within milliseconds by withdrawals and refills at the best quotes.</p>
      <img src="assets/hawkes.svg" alt="Key Hawkes excitation effects"></div>
    <div class="card"><h3>Volume builds into the close</h3>
      <p>Traded volume rises through the afternoon, when liquidity is cheapest to access.</p>
      <img src="assets/intraday_volume.svg" alt="Intraday volume"></div>
  </div>

  <h2>Selected figures from the study</h2>
  <div class="figs">
{figs}
  </div>
</main>

<footer class="wrap">
  <p>Source data: Refinitiv (LSEG) Tick History exports, used under an academic licence. Only aggregate results and charts are published. Every number on this page comes from the same reproducible pipeline and results as the study; a version run end to end on synthetic data is at <a href="../synthetic/index.html">/synthetic/</a>.</p>
  <p>© 2026 Lungelo Phewa · <a href="https://github.com/LungeloPhewa/jse-market-microstructure">Source on GitHub</a></p>
</footer>
</body>
</html>
"""


if __name__ == "__main__":
    R = json.loads((DASH / "data" / "results.json").read_text())
    build_charts(R)
    (DASH / "index.html").write_text(page(R))
    print("dashboard written to", DASH)
