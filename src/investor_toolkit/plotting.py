"""Matplotlib figures with a consistent, colour-vision-deficiency-safe style.

Categorical colours are assigned in a fixed order and follow the series name,
so the same portfolio keeps its colour across figures. Multi-series charts
carry both a legend and direct end labels.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE_RAMP = {100: "#cde2fb", 250: "#86b6ef", 450: "#2a78d6", 550: "#1c5cab", 700: "#0d366b"}
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a52a2a"]
)


def apply_style() -> None:
    """Set global Matplotlib defaults for this package's figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "axes.prop_cycle": plt.cycler(color=SERIES),
        }
    )


def colors_for(names: Sequence[str]) -> dict[str, str]:
    """Fixed colour per name, in slot order (at most eight series)."""
    if len(names) > len(SERIES):
        raise ValueError("At most eight categorical series; fold the rest into 'other'")
    return {name: SERIES[i] for i, name in enumerate(names)}


def _end_labels(ax, series: dict[str, pd.Series], fmt=None, min_gap: float = 0.05) -> None:
    """Label each line at its last point, nudging labels apart vertically."""
    y0, y1 = ax.get_ylim()
    log = ax.get_yscale() == "log"

    def to_axes(y):
        return (np.log(y) - np.log(y0)) / (np.log(y1) - np.log(y0)) if log else (y - y0) / (y1 - y0)

    items = sorted(
        ((to_axes(s.dropna().iloc[-1]), name, s.dropna().iloc[-1]) for name, s in series.items()),
        key=lambda item: item[0],
    )
    placed = []
    for pos, name, value in items:
        if placed and pos - placed[-1] < min_gap:
            pos = placed[-1] + min_gap
        placed.append(pos)
        text = name if fmt is None else f"{name} {fmt(value)}"
        ax.annotate(
            text,
            xy=(1.0, pos),
            xycoords=("axes fraction", "axes fraction"),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )


def plot_lines(
    data: pd.DataFrame,
    ax=None,
    log: bool = False,
    title: str | None = None,
    ylabel: str | None = None,
    label_fmt=None,
    percent: bool = False,
    legend_loc: str = "best",
):
    """Line chart with legend and direct end labels; colours follow column order."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.2))
    palette = colors_for(list(data.columns))
    for column in data.columns:
        ax.plot(data.index, data[column], color=palette[column], label=column)
    if log:
        ax.set_yscale("log")
        plain = plt.FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
    if percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if len(data.columns) > 1:
        ax.legend(loc=legend_loc, ncols=min(len(data.columns), 3))
        _end_labels(ax, {c: data[c] for c in data.columns}, label_fmt)
    return ax


def plot_return_distribution(returns: pd.Series, normal, student_t, axes=None, title: str = ""):
    """Histogram with fitted normal and Student-t densities, plus a normal Q-Q plot."""
    from investor_toolkit.returns import qq_points

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    r = returns.dropna().to_numpy()
    hist_ax, qq_ax = axes
    hist_ax.hist(r, bins=60, density=True, color=BLUE_RAMP[250], edgecolor=SURFACE, linewidth=0.8)
    grid = np.linspace(r.min(), r.max(), 400)
    hist_ax.plot(grid, normal.frozen().pdf(grid), color=SERIES[1], label="Normal fit")
    hist_ax.plot(grid, student_t.frozen().pdf(grid), color=SERIES[2], label="Student-t fit")
    hist_ax.set_yscale("log")
    hist_ax.set_ylim(bottom=max(1e-3, hist_ax.get_ylim()[0]))
    hist_ax.set_title(f"{title} - density (log scale)".strip(" -"))
    hist_ax.set_xlabel("Return")
    hist_ax.legend()

    theoretical, sample = qq_points(r)
    qq_ax.plot(theoretical, sample, "o", markersize=3, color=SERIES[0], alpha=0.6)
    lims = [min(theoretical.min(), sample.min()), max(theoretical.max(), sample.max())]
    qq_ax.plot(lims, lims, color=INK_MUTED, linewidth=1.0)
    qq_ax.set_title("Q-Q plot against fitted normal")
    qq_ax.set_xlabel("Normal quantile")
    qq_ax.set_ylabel("Observed quantile")
    return axes


def plot_correlation(corr: pd.DataFrame, ax=None, title: str = "Correlation"):
    """Correlation heatmap on a diverging blue-gray-red scale fixed to [-1, 1]."""
    if ax is None:
        size = 1.0 + 0.7 * len(corr)
        _, ax = plt.subplots(figsize=(size + 1.2, size))
    image = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.6 else INK,
            )
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    return ax


def plot_frontier(
    frontier: pd.DataFrame,
    assets: pd.DataFrame,
    portfolios: pd.DataFrame | None = None,
    ax=None,
    title: str = "Efficient frontier (annualised)",
    label_offsets: dict | None = None,
):
    """Efficient frontier with individual assets and optional highlighted portfolios.

    ``assets`` and ``portfolios`` have columns ``volatility`` and ``mean`` and are
    labelled by their index. ``label_offsets`` maps a name to an ``(x, y)`` label
    offset in points, to separate the labels of nearby points.
    """
    offsets = label_offsets or {}
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frontier["volatility"], frontier["mean"], color=SERIES[0], label="Long-only frontier")
    ax.scatter(
        assets["volatility"],
        assets["mean"],
        s=36,
        color=INK_MUTED,
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1.5,
        label="Single assets",
    )
    for name, row in assets.iterrows():
        ax.annotate(
            name,
            (row["volatility"], row["mean"]),
            xytext=offsets.get(name, (5, 4)),
            textcoords="offset points",
            fontsize=8,
            color=INK_SECONDARY,
        )
    if portfolios is not None:
        ax.scatter(
            portfolios["volatility"],
            portfolios["mean"],
            s=48,
            color=SERIES[1],
            zorder=4,
            edgecolor=SURFACE,
            linewidth=1.5,
            marker="D",
            label="Portfolios",
        )
        for name, row in portfolios.iterrows():
            ax.annotate(
                name,
                (row["volatility"], row["mean"]),
                xytext=offsets.get(name, (5, -10)),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected return")
    ax.set_xlim(left=0)
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


def plot_fan(percentiles: pd.DataFrame, ax=None, title: str = "", ylabel: str = "Wealth"):
    """Fan chart of outcome percentiles over time (columns: 5, 25, 50, 75, 95)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.2))
    x = percentiles.index
    ax.fill_between(
        x,
        percentiles[5],
        percentiles[95],
        color=BLUE_RAMP[100],
        linewidth=0,
        label="5th-95th percentile",
    )
    ax.fill_between(
        x,
        percentiles[25],
        percentiles[75],
        color=BLUE_RAMP[250],
        linewidth=0,
        label="25th-75th percentile",
    )
    ax.plot(x, percentiles[50], color=BLUE_RAMP[700], label="Median")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    return ax


def plot_coefficients(
    summary: pd.DataFrame, ax=None, title: str = "Factor exposures", level: float = 1.96
):
    """Horizontal bars of regression coefficients with confidence whiskers.

    ``summary`` has columns ``coef`` and ``std_err`` (e.g.
    :meth:`RegressionResult.summary`).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.45 * len(summary) + 1.2))
    y = np.arange(len(summary))
    ax.barh(y, summary["coef"], height=0.55, color=SERIES[0], edgecolor=SURFACE)
    ax.errorbar(
        summary["coef"],
        y,
        xerr=level * summary["std_err"],
        fmt="none",
        ecolor=INK_SECONDARY,
        elinewidth=1.0,
        capsize=3,
    )
    ax.axvline(0, color=AXIS, linewidth=1.0)
    ax.set_yticks(y, summary.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    return ax


def plot_convergence(
    sizes,
    errors,
    ax=None,
    title: str = "Monte Carlo error vs number of paths",
    reference_slope: float = -0.5,
):
    """Log-log plot of estimation error with a reference power-law slope."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sizes = np.asarray(sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    slope = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
    ax.loglog(sizes, errors, "o", color=SERIES[0], markersize=7, label="Observed RMS error")
    ref = np.exp(np.log(errors[0]) + reference_slope * (np.log(sizes) - np.log(sizes[0])))
    ax.loglog(
        sizes, ref, color=INK_MUTED, linewidth=1.0, label=f"Slope {reference_slope:g} reference"
    )
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("RMS error")
    ax.set_title(f"{title}\nfitted slope {slope:.2f}")
    ax.legend()
    return ax
