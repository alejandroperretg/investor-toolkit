"""Generate the inline SVG charts of the project page on alejandroperretg.github.io.

The charts are drawn from the toolkit's own output and styled only through CSS classes
defined in the site's stylesheet, so they follow its light and dark themes:

* ``ax`` axis lines, ``ln`` main line, ``h`` dashed accent line, ``hb`` shaded band
* ``pt`` accent dot, ``pt-ink`` ink dot, ``bar`` ink bar, ``bar2`` accent bar
* ``tk`` small muted label, ``tv`` bold value label

Each chart replaces the content between ``<!-- chart:NAME -->`` and ``<!-- /chart:NAME -->``
markers in the given HTML files::

    uv run python scripts/site_charts.py <site-repo>/pages/en/investor-toolkit.html \
        <site-repo>/pages/es/investor-toolkit.html <site-repo>/pages/en/index.html \
        <site-repo>/pages/es/index.html
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from investor_toolkit import montecarlo as mc
from investor_toolkit import proxies, universe
from investor_toolkit.tax import GermanTax, blended_partial_exemption

N_PATHS = 10_000


def _k(value: float) -> str:
    return f"€{value / 1000:.0f}k"


def _path(points) -> str:
    return "M" + "L".join(f"{x:.2f} {y:.2f}" for x, y in points)


def fan(long: pd.DataFrame, compact: bool) -> str:
    """Savings fan chart: 30 years saving, then 30 years of 4% withdrawals."""
    balanced = 0.6 * long["world_equity"] + 0.4 * long["bund_10y"]
    plan = mc.SavingsPlan(
        monthly_contribution=500,
        accumulation_years=30,
        withdrawal_years=30,
        withdrawal_rate=0.04,
        annual_fee=0.002,
        tax=GermanTax(partial_exemption=blended_partial_exemption(0.6)),
    )
    result = mc.simulate(
        mc.Bootstrap(balanced, long["inflation"], mean_block=12), plan, n_paths=N_PATHS, seed=1
    )
    pct = result.percentiles()
    x0, x1, y0, y1, ymax = 9.0, 77.0, 52.0, 6.0, 800_000

    def xy(t, v):
        return x0 + (x1 - x0) * t / 60, y0 - (y0 - y1) * min(v, ymax) / ymax

    def line(col):
        return _path(xy(t, v) for t, v in zip(pct.index, pct[col], strict=True))

    band = (
        _path(
            [xy(t, v) for t, v in zip(pct.index, pct[95], strict=True)]
            + [xy(t, v) for t, v in zip(pct.index[::-1], pct[5][::-1], strict=True)]
        )
        + "Z"
    )
    median = pct.loc[30, 50]
    out = [
        '<svg viewBox="0 0 80 60" role="img" aria-label="Simulated wealth over 60 years: '
        f'median about {median:,.0f} euros at retirement, 5 to 95 percent range as a band">',
        f'<path class="hb" d="{band}"/>',
    ]
    out += [f'<path class="h" d="{line(col)}"/>' for col in (5, 25, 75, 95)]
    out.append(f'<path class="ax" d="M{x0} {y1 - 2}L{x0} {y0}L{x1} {y0}"/>')
    rx = xy(30, 0)[0]
    out.append(f'<path class="ax" d="M{rx:.2f} {y0}L{rx:.2f} {y1}"/>')
    for yr in (0, 30, 60):
        out.append(
            f'<text class="tk" x="{xy(yr, 0)[0]:.2f}" y="{y0 + 4.2}" text-anchor="middle">{yr}</text>'
        )
    for v in (0, 400_000, 800_000):
        out.append(
            f'<text class="tk" x="{x0 - 1.5}" y="{xy(0, v)[1] + 0.8:.2f}" text-anchor="end">{_k(v)}</text>'
        )
    out.append(f'<text class="tk" x="{rx + 1:.2f}" y="{y1 + 2}">retire at year 30</text>')
    out.append(f'<path class="ln" d="{line(50)}"/>')
    mx, my = xy(30, median)
    out.append(f'<circle class="pt" cx="{mx:.2f}" cy="{my:.2f}" r=".9"/>')
    if not compact:
        out.append(
            f'<text class="tk" x="{mx - 1.5:.2f}" y="{my - 2:.2f}" text-anchor="end">median {_k(median)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def convergence() -> str:
    """Relative error of the simulated mean vs number of paths, against the 1/sqrt(N) law."""
    model = mc.GBM(expected_return=0.07, volatility=0.15)
    plan = mc.SavingsPlan(
        monthly_contribution=100, contributions_indexed=False, accumulation_years=20
    )
    exact, _ = mc.gbm_wealth_moments(model, 100, 240)
    sizes = np.array([100, 400, 1600, 6400, 25600])
    rms = np.array(
        [
            np.sqrt(
                np.mean(
                    [
                        (mc.simulate(model, plan, int(n), seed=s).wealth[:, -1].mean() - exact) ** 2
                        for s in range(30)
                    ]
                )
            )
            / exact
            for n in sizes
        ]
    )
    slope = np.polyfit(np.log(sizes), np.log(rms), 1)[0]
    x0, x1, y0, y1 = 12.0, 76.0, 42.0, 5.0

    def xy(n, e):
        return (
            x0 + (x1 - x0) * (np.log10(n) - 2) / (np.log10(30_000) - 2),
            y0 - (y0 - y1) * (np.log10(e) - np.log10(2e-3)) / (np.log10(0.1) - np.log10(2e-3)),
        )

    ref = [xy(n, rms[0] * (n / sizes[0]) ** -0.5) for n in (100, 30_000)]
    out = [
        f'<svg viewBox="0 0 80 50" role="img" aria-label="Monte Carlo error falls with slope '
        f'{slope:.2f}, against the theoretical minus one half">',
        f'<path class="ax" d="M{x0} {y1 - 1}L{x0} {y0}L{x1} {y0}"/>',
        f'<path class="h" d="{_path(ref)}"/>',
    ]
    out += [
        f'<circle class="pt" cx="{x:.2f}" cy="{y:.2f}" r="1"/>'
        for x, y in (xy(n, e) for n, e in zip(sizes, rms, strict=True))
    ]
    for n, label in ((100, "10²"), (1000, "10³"), (10_000, "10⁴")):
        out.append(
            f'<text class="tk" x="{xy(n, 1e-2)[0]:.2f}" y="{y0 + 4}" text-anchor="middle">{label}</text>'
        )
    for e, label in ((2e-3, "0.2%"), (1e-2, "1%"), (1e-1, "10%")):
        out.append(
            f'<text class="tk" x="{x0 - 1.5}" y="{xy(100, e)[1] + 0.8:.2f}" text-anchor="end">{label}</text>'
        )
    out.append(f'<text class="tk" x="{x1}" y="{y0 + 4}" text-anchor="end">paths</text>')
    lx, ly = x0 + 4, y0 - 9
    out += [
        f'<circle class="pt" cx="{lx + 1.5:.2f}" cy="{ly - 0.7:.2f}" r="1"/>',
        f'<text class="tk" x="{lx + 4:.2f}" y="{ly:.2f}">simulation, fitted slope {slope:.2f}</text>',
        f'<path class="h" d="M{lx:.2f} {ly + 3.3:.2f}L{lx + 3:.2f} {ly + 3.3:.2f}"/>',
        f'<text class="tk" x="{lx + 4:.2f}" y="{ly + 4:.2f}">theory, slope -1/2</text>',
        "</svg>",
    ]
    return "".join(out)


def futures(long: pd.DataFrame) -> str:
    """100 representative after-tax outcomes of a €500/month savings plan."""
    balanced = 0.6 * long["world_equity"] + 0.4 * long["bund_10y"]
    plan = mc.SavingsPlan(
        monthly_contribution=500,
        accumulation_years=30,
        annual_fee=0.002,
        tax=GermanTax(partial_exemption=blended_partial_exemption(0.6)),
    )
    result = mc.simulate(
        mc.Bootstrap(balanced, long["inflation"], mean_block=12), plan, n_paths=N_PATHS, seed=1
    )
    after_tax = result.terminal(after_tax=True)
    values = np.quantile(after_tax, (np.arange(100) + 0.5) / 100)
    paid, median = 180_000, float(np.median(after_tax))
    lo, hi, width = 0, 700_000, 25_000
    x0, x1, base = 4.0, 77.0, 31.0

    def x(v):
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)

    below = int((values < paid).sum())
    out = [
        f'<svg viewBox="0 0 80 38" role="img" aria-label="100 simulated after-tax outcomes of a 500 euro '
        f'per month savings plan: median {median:,.0f} euros, {below} of 100 below the 180,000 euros paid in">',
        f'<path class="ax" d="M{x0} {base + 1.4}L{x1} {base + 1.4}"/>',
    ]
    for v in range(100_000, 700_000, 100_000):
        out.append(
            f'<text class="tk" x="{x(v):.2f}" y="{base + 4.6}" text-anchor="middle">{_k(v)}</text>'
        )
    for level, label, anchor, dx in (
        (paid, f"paid in {_k(paid)}", "end", -0.8),
        (median, f"median {_k(median)}", "start", 0.8),
    ):
        out.append(f'<path class="ax" d="M{x(level):.2f} {base + 1.4}L{x(level):.2f} 3"/>')
        out.append(
            f'<text class="tk" x="{x(level) + dx:.2f}" y="4.6" text-anchor="{anchor}">{label}</text>'
        )
    stack: dict[int, int] = {}
    for v in values:
        b = int((v - lo) // width)
        k = stack.get(b, 0)
        stack[b] = k + 1
        cls = "pt-ink" if v < paid else "pt"
        out.append(
            f'<circle class="{cls}" cx="{x(lo + (b + 0.5) * width):.2f}" cy="{base - k * 2.45:.2f}" r="1.05"/>'
        )
    out.append("</svg>")
    return "".join(out)


def crash(long: pd.DataFrame) -> str:
    """€10,000 invested at the August 2000 peak, nominal and in today's money."""
    growth = (1 + long["world_equity"]).cumprod()
    price_level = (1 + long["inflation"]).cumprod()
    peak = growth.loc[:"2001"].idxmax()
    nominal = 10_000 * growth.loc[peak:] / growth[peak]
    real = nominal / (price_level.loc[peak:] / price_level[peak])
    t0, t1 = nominal.index[0], nominal.index[-1]
    x0, x1, y0, y1, ymax = 9.0, 76.0, 34.0, 4.0, 55_000

    def xy(d, v):
        return x0 + (x1 - x0) * (d - t0).days / (t1 - t0).days, y0 - (y0 - y1) * v / ymax

    def recovered(series):
        trough = series.loc[:"2004"].idxmin()
        return trough, series.loc[trough:][series.loc[trough:] >= 10_000].index[0]

    trough, back_nominal = recovered(nominal)
    _, back_real = recovered(real)
    out = [
        f'<svg viewBox="0 0 80 40" role="img" aria-label="10,000 euros invested at the August 2000 peak '
        f"fell to {nominal[trough]:,.0f} in {trough:%B %Y}; back to 10,000 in {back_nominal:%B %Y}, or "
        f"{back_real:%B %Y} after inflation; worth {nominal.iloc[-1]:,.0f} in {t1:%Y}, "
        f"{real.iloc[-1]:,.0f} in today's money\">",
        f'<path class="ax" d="M{x0} {y1 - 1}L{x0} {y0}L{x1} {y0}"/>',
        f'<path class="ax" d="M{x0} {xy(t0, 10_000)[1]:.2f}L{x1} {xy(t0, 10_000)[1]:.2f}"/>',
    ]
    for v in (0, 10_000, 25_000, 50_000):
        out.append(
            f'<text class="tk" x="{x0 - 1.2}" y="{xy(t0, v)[1] + 0.8:.2f}" text-anchor="end">{_k(v)}</text>'
        )
    for year in (2005, 2010, 2015, 2020, 2025):
        out.append(
            f'<text class="tk" x="{xy(pd.Timestamp(year, 1, 1), 0)[0]:.2f}" y="{y0 + 4}" '
            f'text-anchor="middle">{year}</text>'
        )
    out.append(f'<path class="h" d="{_path(xy(d, v) for d, v in real.items())}"/>')
    out.append(f'<path class="ln" d="{_path(xy(d, v) for d, v in nominal.items())}"/>')
    for d, series, label, anchor, label_v in (
        (trough, nominal, f"{trough:%b %Y}: €{nominal[trough]:,.0f}", "start", 24_000),
        (back_nominal, nominal, f"back to €10k: {back_nominal:%b %Y}", "end", 31_000),
    ):
        px, py = xy(d, series[d])
        ly = xy(d, label_v)[1]
        tx = px - 0.6 if anchor == "start" else px + 0.6
        out += [
            f'<path class="ax" d="M{px:.2f} {py - 1.1:.2f}L{px:.2f} {ly + 0.9:.2f}"/>',
            f'<circle class="pt" cx="{px:.2f}" cy="{py:.2f}" r=".9"/>',
            f'<text class="tv" x="{tx:.2f}" y="{ly:.2f}" text-anchor="{anchor}">{label}</text>',
        ]
    lx = x0 + 3
    out += [
        f'<path class="ln" d="M{lx} 3.4L{lx + 3} 3.4"/>',
        f'<text class="tk" x="{lx + 4}" y="4.1">nominal</text>',
        f'<path class="h" d="M{lx + 18} 3.4L{lx + 21} 3.4"/>',
        f'<text class="tk" x="{lx + 22}" y="4.1">today\'s money</text>',
        "</svg>",
    ]
    return "".join(out)


def bonds(long: pd.DataFrame) -> str:
    """Calendar-year returns of world stocks and 10-year Bunds in 2002, 2008 and 2022."""
    calendar = (1 + long[["world_equity", "bund_10y"]]).groupby(long.index.year).prod() - 1
    years = [2002, 2008, 2022]
    x0, x1, zero, scale = 6.0, 77.0, 15.0, 34.0
    out = [
        '<svg viewBox="0 0 80 40" role="img" aria-label="Calendar-year returns in euros: in 2002 and '
        '2008 stocks fell while German government bonds rose; in 2022 both fell">',
        f'<path class="ax" d="M{x0} {zero}L{x1} {zero}"/>',
    ]
    group = (x1 - x0) / len(years)
    for i, year in enumerate(years):
        cx = x0 + group * (i + 0.5)
        for j, (column, cls) in enumerate((("world_equity", "bar"), ("bund_10y", "bar2"))):
            r = calendar.loc[year, column]
            bx = cx + (j - 1) * 6.2 + 0.4
            top, height = (zero - r * scale, r * scale) if r > 0 else (zero, -r * scale)
            label_y = zero - r * scale - 1.0 if r > 0 else zero - r * scale + 2.8
            out += [
                f'<rect class="{cls}" x="{bx:.2f}" y="{top:.2f}" width="5.4" height="{height:.2f}"/>',
                f'<text class="tv" x="{bx + 2.7:.2f}" y="{label_y:.2f}" text-anchor="middle">{r:+.0%}</text>',
            ]
        out.append(f'<text class="tk" x="{cx:.2f}" y="38.6" text-anchor="middle">{year}</text>')
    out += [
        f'<rect class="bar" x="{x0 + 1}" y="1.6" width="2" height="2"/>',
        f'<text class="tk" x="{x0 + 4}" y="3.3">world stocks</text>',
        f'<rect class="bar2" x="{x0 + 24}" y="1.6" width="2" height="2"/>',
        f'<text class="tk" x="{x0 + 27}" y="3.3">10-year Bunds</text>',
        "</svg>",
    ]
    return "".join(out)


def spread() -> str:
    """Typical yearly swing of one stock, five stocks and the world index."""
    returns = universe.load_asset_returns()
    stocks = returns[universe.MODEL_PORTFOLIOS["concentrated"].tickers]
    rows = [
        ("one German stock", (stocks.std() * np.sqrt(12)).mean()),
        ("five German stocks", stocks.mean(axis=1).std() * np.sqrt(12)),
        ("world index", returns["EUNL.DE"].std() * np.sqrt(12)),
    ]
    x0, x1 = 24.0, 70.0
    out = [
        '<svg viewBox="0 0 80 22" role="img" aria-label="Typical yearly swing: '
        + ", ".join(f"{label} {v:.0%}" for label, v in rows)
        + '">'
    ]
    for i, (label, v) in enumerate(rows):
        y = 2 + i * 6.5
        w = (x1 - x0) * v / 0.30
        out += [
            f'<text class="tk" x="{x0 - 1.5}" y="{y + 2.9:.2f}" text-anchor="end">{label}</text>',
            f'<rect class="bar2" x="{x0}" y="{y:.2f}" width="{w:.2f}" height="4"/>',
            f'<text class="tv" x="{x0 + w + 1.2:.2f}" y="{y + 2.9:.2f}">{v:.0%}</text>',
        ]
    out.append("</svg>")
    return "".join(out)


def withdrawals(long: pd.DataFrame) -> str:
    """Two 10x10 grids: histories in which 4% and 3% withdrawals last 30 years."""
    balanced = 0.6 * long["world_equity"] + 0.4 * long["bund_10y"]
    model = mc.Bootstrap(balanced, long["inflation"], mean_block=12)
    outcomes = []
    for rate in (0.04, 0.03):
        plan = mc.SavingsPlan(
            initial=500_000,
            monthly_contribution=0,
            accumulation_years=0,
            withdrawal_years=30,
            withdrawal_rate=rate,
            annual_fee=0.002,
        )
        outcomes.append(
            (rate, round(mc.simulate(model, plan, n_paths=N_PATHS, seed=2).success_rate() * 100))
        )
    out = [
        '<svg viewBox="0 0 80 40" role="img" aria-label="'
        + "; ".join(
            f"withdrawing {r:.0%} of 500,000 euros: {k} of 100 histories last 30 years"
            for r, k in outcomes
        )
        + '">'
    ]
    for g, (rate, lasted) in enumerate(outcomes):
        gx = 4 + g * 40
        out += [
            f'<text class="tv" x="{gx}" y="3.4">{rate:.0%}: €{500_000 * rate:,.0f} a year</text>',
            f'<text class="tk" x="{gx}" y="6.6">{lasted} of 100 last 30 years</text>',
        ]
        for k in range(100):
            row, col = divmod(k, 10)
            cls = "bar2" if k < lasted else "bar"
            out.append(
                f'<rect class="{cls}" x="{gx + col * 3.1:.2f}" y="{9 + row * 3.1:.2f}" width="2.4" height="2.4"/>'
            )
    out.append("</svg>")
    return "".join(out)


def build_all() -> dict[str, str]:
    long = proxies.long_history()
    return {
        "fan-card": fan(long, compact=True),
        "fan-page": fan(long, compact=False),
        "convergence": convergence(),
        "futures": futures(long),
        "crash": crash(long),
        "bonds": bonds(long),
        "spread": spread(),
        "withdrawals": withdrawals(long),
    }


def inject(html: str, charts: dict[str, str]) -> tuple[str, list[str]]:
    """Replace every ``<!-- chart:NAME -->...<!-- /chart:NAME -->`` block with the new SVG."""
    replaced = []
    for name, svg in charts.items():
        pattern = re.compile(
            rf"(<!-- chart:{re.escape(name)} -->).*?(<!-- /chart:{re.escape(name)} -->)", re.S
        )
        html, count = pattern.subn(lambda m, svg=svg: m.group(1) + svg + m.group(2), html)
        if count:
            replaced.append(name)
    return html, replaced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="+", type=Path, help="HTML files containing chart markers")
    args = parser.parse_args()
    charts = build_all()
    for page in args.pages:
        html, replaced = inject(page.read_text(encoding="utf-8"), charts)
        page.write_text(html, encoding="utf-8", newline="\n")
        print(f"{page}: {', '.join(replaced) or 'no chart markers found'}")


if __name__ == "__main__":
    main()
