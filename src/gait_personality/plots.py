"""Static figures for the report (matplotlib, light background).

Colors: the baseline ("standard") is neutral gray; the three personalities use
the first three slots of a validated categorical palette (blue, orange, aqua),
which stay distinguishable for color-vision deficiencies in scatter plots.
Every point and line is also direct-labeled, so identity never relies on color alone.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from gait_personality.evaluation import METRIC_INFO
from gait_personality.styles import STYLES

STYLE_COLORS = {
    "standard": "#7a7974",  # neutral gray = baseline
    "serving": "#2a78d6",  # blue
    "eco": "#eb6834",  # orange
    "energetic": "#1baf7a",  # aqua
}
STYLE_MARKERS = {"standard": "o", "serving": "s", "eco": "^", "energetic": "D"}
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def _style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _label(style: str) -> str:
    return STYLES[style].label if style in STYLES else style


def tradeoff_plot(summary: pd.DataFrame, y_metrics: Sequence[str] = ("vel_err", "rp_rate_rms")):
    """Small multiples sharing the x axis (cost of transport)."""
    fig, axes = plt.subplots(1, len(y_metrics), figsize=(5.2 * len(y_metrics), 4.2), sharex=True)
    if len(y_metrics) == 1:
        axes = [axes]
    for ax, ym in zip(axes, y_metrics):
        for _, r in summary.iterrows():
            st = r["style"]
            ax.errorbar(
                r["cot"], r[ym],
                xerr=r.get("cot_ci"), yerr=r.get(f"{ym}_ci"),
                fmt=STYLE_MARKERS.get(st, "o"), markersize=9,
                color=STYLE_COLORS.get(st, TEXT_2), ecolor=STYLE_COLORS.get(st, TEXT_2),
                elinewidth=1.2, capsize=3, markeredgecolor="white", markeredgewidth=1.5,
                label=_label(st),
            )
            ax.annotate(_label(st), (r["cot"], r[ym]), xytext=(8, 6), textcoords="offset points",
                        fontsize=9, color=TEXT)
        _style_axes(ax)
        ax.set_xlabel(METRIC_INFO["cot"][0] + "  (lower = more efficient)")
        ax.set_ylabel(METRIC_INFO[ym][0] + "  (lower = better)")
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Trade-offs between gait personalities (mean, 95% CI)", color=TEXT, fontsize=12)
    fig.tight_layout()
    return fig


def metric_bars(summary: pd.DataFrame, metrics: Optional[Sequence[str]] = None, ncols: int = 3):
    """One small panel per metric, one bar per style (fixed color order)."""
    metrics = list(metrics or ["cot", "vel_err", "rp_rate_rms", "tilt_rms_deg", "height_std_cm", "foot_lift_cm"])
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows))
    axes = axes.ravel()
    styles = list(summary["style"])
    for ax, m in zip(axes, metrics):
        vals = summary[m].to_numpy()
        cis = summary.get(f"{m}_ci")
        ax.bar(
            range(len(styles)), vals,
            yerr=None if cis is None else cis.to_numpy(),
            color=[STYLE_COLORS.get(s, TEXT_2) for s in styles],
            edgecolor="white", linewidth=2, width=0.72,
            error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": TEXT_2},
        )
        ax.set_xticks(range(len(styles)))
        ax.set_xticklabels([_label(s) for s in styles], fontsize=9)
        label, lower_better = METRIC_INFO[m]
        ax.set_title(label + ("  ↓" if lower_better else "  ↑"), fontsize=10, color=TEXT)
        _style_axes(ax)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def training_curves(histories: Dict[str, List[dict]]):
    """Evaluation reward vs. environment steps for each training run."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    plotted = False
    for name, hist in histories.items():
        if not hist:
            continue
        plotted = True
        x = [h["step"] / 1e6 for h in hist]
        y = [h["reward"] for h in hist]
        st = name.split("/")[-1]
        color = STYLE_COLORS.get(st, TEXT_2)
        ax.plot(x, y, color=color, linewidth=1.8, marker=STYLE_MARKERS.get(st, "o"), markersize=5, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=TEXT, va="center")
    _style_axes(ax)
    ax.set_xlabel("Environment steps [million]")
    ax.set_ylabel("Evaluation episode reward")
    if plotted:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
