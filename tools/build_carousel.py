"""Build the LinkedIn carousel (a multi-page PDF, 1080x1350 per slide).

Summaries only: results and charts, no code. Uses the same aggregate file and
chart functions as the dashboard.

    python tools/build_carousel.py [output.pdf]
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_dashboard import COL, DASH, INK, MUTED, build_charts  # noqa: E402

SITE = "lungelophewa.github.io/jse-market-microstructure"


def img(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def slides(R, C):
    sp, sq, rho, eff = R["spread_stats"], R["sqrt_law"], R["hawkes_rho_days"], R["hawkes_effects"]
    v0 = {r: R["seasonality"][r]["volatility"]["y"][0] for r in ("AGLJ.J", "GRTJ.J")}
    hl = {r: 1000 * eff["half_life_buy_mo_to_agg_ask_lo"][r] for r in ("AGLJ.J", "GRTJ.J")}
    n = 7

    def foot(i):
        return f'<div class="foot"><span>Lungelo Phewa · JSE market microstructure</span><span>{i} / {n}</span></div>'

    def chart_slide(i, kicker, title, body, chart):
        return f"""<section><div class="kicker">{kicker}</div><h2>{title}</h2><p class="lead">{body}</p>
<div class="chart"><img src="{img(C[chart])}"></div>{foot(i)}</section>"""

    return [
        f"""<section class="cover"><div class="kicker light">Research project · 2026</div>
<h1>What does it really cost to trade a share on the JSE?</h1>
<p class="sub">A study of Anglo American and Growthpoint from one month of tick data: every trade and every change of a ten-level order book.</p>
<div class="stats">
  <div><b>{R['n_trades'] / 1000:,.0f}k</b><span>trades</span></div>
  <div><b>{R['n_quotes'] / 1e6:.1f}m</b><span>quote updates</span></div>
  <div><b>{R['hawkes_n_days']['AGLJ.J'] + R['hawkes_n_days']['GRTJ.J']}</b><span>stock-days modelled</span></div>
</div>
<div class="legend"><span><i style="background:{COL['AGLJ.J']}"></i>Anglo American (AGLJ.J)</span><span><i style="background:{COL['GRTJ.J']}"></i>Growthpoint (GRTJ.J)</span></div>
<div class="swipe">Five findings →</div>{foot(1)}</section>""",
        chart_slide(2, "1 · The cost of a round trip", "Same one-cent tick, twice the cost",
                    f"Growthpoint's effective spread is <b>{sp['GRTJ.J']['effective_bps']:.1f} bps</b> against <b>{sp['AGLJ.J']['effective_bps']:.1f} bps</b> for Anglo, because its tick is about ten times larger relative to its price. {sp['AGLJ.J']['impact_bps'] / sp['AGLJ.J']['effective_bps']:.0%} of Anglo's spread is lost to better-informed traders within a minute.",
                    "spreads"),
        chart_slide(3, "2 · Trade size", "Size matters far less than expected",
                    f"The price impact of a single trade grows with size at an exponent of only <b>{sq['AGLJ.J']['slope']:.2f}</b> and <b>{sq['GRTJ.J']['slope']:.2f}</b>, far below the square-root law's 0.5 (dashed). Large orders are split, so one print reveals little.",
                    "size"),
        chart_slide(4, "3 · Timing", "When you trade matters more",
                    f"Volatility in the first ten minutes is <b>{v0['AGLJ.J']:.1f}×</b> (Anglo) and <b>{v0['GRTJ.J']:.1f}×</b> (Growthpoint) its daily average, and spreads are widest too. An execution schedule that ignores the open overpays.",
                    "intraday_volatility"),
        chart_slide(5, "4 · Order flow", "The order book reacts to itself, in milliseconds",
                    f"A ten-variate Hawkes model fitted to every day finds near-critical self-excitation (median spectral radius <b>{rho['AGLJ.J'][1]:.2f}</b>): most orders are reactions to earlier orders. After a buy, the ask side is refilled with a half-life of about <b>{hl['AGLJ.J']:.0f} ms</b>.",
                    "hawkes"),
        f"""<section><div class="kicker">5 · How it was done</div><h2>From raw exchange data to decisions</h2>
<ol class="steps">
  <li><b>Clean the raw feed.</b> Vendor tick data for trades, quotes and depth, merged into one time-ordered record and every trade signed as buyer- or seller-initiated.</li>
  <li><b>Measure costs.</b> Effective, quoted and realised spreads, price impact, intraday patterns and how fast liquidity returns after large trades.</li>
  <li><b>Model the order flow.</b> A self-exciting point process estimated by maximum likelihood on each day, with goodness-of-fit diagnostics.</li>
  <li><b>Test against known answers.</b> The same pipeline run on a simulated market with known parameters, which caught a silent error that produced plausible but wrong results.</li>
</ol>
<p class="note">Fully reproducible: automated tests, a locked environment, and results rebuilt and published on every update.</p>{foot(6)}</section>""",
        f"""<section class="cover end"><div class="kicker light">Explore the results</div>
<h1>Interactive results and the full study</h1>
<div class="links">
  <div><span>Results dashboard</span><b>{SITE}/dashboard</b></div>
  <div><span>Full study</span><b>{SITE}</b></div>
</div>
<p class="sub">I'm a mathematical statistician looking for graduate quantitative roles in execution, market microstructure, trading analytics and risk. I'd welcome a conversation.</p>
<p class="sig">Lungelo Phewa · MSc Statistics &amp; Data Science (UCT) · MSc Financial Engineering (WQU)</p>{foot(7)}</section>""",
    ]


CSS = f"""
@page {{ size: 1080px 1350px; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Inter, sans-serif; color: {INK}; }}
section {{ width: 1080px; height: 1350px; padding: 90px 80px 0; position: relative; page-break-after: always; background: #f5f7fb; overflow: hidden; }}
.kicker {{ font-size: 24px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: {COL['AGLJ.J']}; }}
.kicker.light {{ color: #9fb3d9; }}
h1 {{ font-size: 78px; line-height: 1.06; font-weight: 800; letter-spacing: -0.02em; margin-top: 28px; }}
h2 {{ font-size: 56px; line-height: 1.1; font-weight: 800; letter-spacing: -0.015em; margin-top: 18px; }}
.lead {{ font-size: 29px; line-height: 1.45; color: #374151; margin-top: 26px; }}
.lead b {{ color: {INK}; }}
.chart {{ position: absolute; left: 60px; right: 60px; bottom: 120px; height: 640px; background: #fff; border-radius: 24px; border: 2px solid #e5e7eb; display: flex; align-items: center; justify-content: center; padding: 24px; }}
.chart img {{ max-width: 100%; max-height: 100%; }}
.foot {{ position: absolute; left: 80px; right: 80px; bottom: 44px; display: flex; justify-content: space-between; font-size: 22px; color: {MUTED}; }}
.cover {{ background: {INK}; color: #fff; }}
.cover .foot {{ color: #9fb3d9; }}
.sub {{ font-size: 32px; line-height: 1.45; color: #d4dcec; margin-top: 34px; }}
.stats {{ display: flex; gap: 24px; margin-top: 70px; }}
.stats div {{ flex: 1; background: rgba(255,255,255,.07); border: 2px solid rgba(255,255,255,.14); border-radius: 22px; padding: 30px 28px; }}
.stats b {{ display: block; font-size: 64px; font-weight: 800; }}
.stats span {{ font-size: 26px; color: #b9c6df; }}
.legend {{ display: flex; gap: 36px; margin-top: 52px; font-size: 26px; color: #d4dcec; }}
.legend i {{ display: inline-block; width: 22px; height: 22px; border-radius: 50%; margin-right: 12px; vertical-align: -3px; }}
.swipe {{ position: absolute; right: 80px; bottom: 120px; font-size: 34px; font-weight: 700; color: #fff; background: {COL['AGLJ.J']}; padding: 18px 30px; border-radius: 999px; }}
.steps {{ margin-top: 44px; padding-left: 0; list-style: none; counter-reset: s; }}
.steps li {{ counter-increment: s; position: relative; padding: 26px 28px 26px 118px; background: #fff; border: 2px solid #e5e7eb; border-radius: 22px; margin-bottom: 20px; font-size: 27px; line-height: 1.42; color: #374151; }}
.steps li b {{ color: {INK}; }}
.steps li::before {{ content: counter(s); position: absolute; left: 30px; top: 26px; width: 62px; height: 62px; border-radius: 50%; background: {INK}; color: #fff; font-weight: 800; font-size: 30px; display: flex; align-items: center; justify-content: center; }}
.note {{ font-size: 26px; color: {MUTED}; margin-top: 18px; line-height: 1.45; }}
.links {{ margin-top: 60px; display: flex; flex-direction: column; gap: 22px; }}
.links div {{ background: #fff; color: {INK}; border-radius: 22px; padding: 28px 34px; }}
.links span {{ display: block; font-size: 24px; color: {MUTED}; font-weight: 600; }}
.links b {{ font-size: 33px; word-break: break-all; }}
.end .sub {{ margin-top: 60px; }}
.sig {{ font-size: 25px; color: #9fb3d9; margin-top: 36px; }}
"""

JS = """const { chromium } = require('playwright');
(async () => { const [src, out] = process.argv.slice(2);
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + src); await p.waitForTimeout(800);
  await p.pdf({ path: out, width: '1080px', height: '1350px', printBackground: true, preferCSSPageSize: true });
  await b.close(); })();"""


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else DASH.parent / "out" / "linkedin-carousel.pdf").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    R = json.loads((DASH / "data" / "results.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        C = build_charts(R, fmt="png", folder=Path(tmp), scale=2.0, size=(8.6, 6.2))
        html = Path(tmp) / "carousel.html"
        html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(slides(R, C))}</body></html>")
        (Path(tmp) / "pdf.js").write_text(JS)
        subprocess.run(["node", str(Path(tmp) / "pdf.js"), str(html), str(out)], check=True,
                       cwd="/tmp")
        (out.with_suffix(".html")).write_text(html.read_text())
    print("carousel written to", out)
