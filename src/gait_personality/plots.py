"""Static figures for the report (matplotlib, light background).

Colors: the baseline ("standard") is neutral gray; the three personalities use
the first three slots of a validated categorical palette (blue, orange, aqua),
which stay distinguishable for color-vision deficiencies in scatter plots.
Every point and line is also direct-labeled, so identity never relies on color alone.

Figures are closed before they are returned: they can still be shown with
``display(fig)`` or saved, but notebooks do not render them a second time.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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
BASE_STYLE = "standard"
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


def _spread(values: Sequence[float], min_gap: float) -> List[float]:
    """Move positions as little as possible so neighbours are >= ``min_gap`` apart.

    Overlapping labels are merged into clusters; each cluster is laid out with
    ``min_gap`` spacing and centered on the mean of its desired positions.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters = [[i] for i in order]

    def layout(cluster: List[int]) -> List[float]:
        center = sum(values[i] for i in cluster) / len(cluster)
        start = center - min_gap * (len(cluster) - 1) / 2
        return [start + k * min_gap for k in range(len(cluster))]

    merged = True
    while merged:
        merged = False
        for k in range(len(clusters) - 1):
            if layout(clusters[k])[-1] + min_gap > layout(clusters[k + 1])[0] + 1e-12:
                clusters[k:k + 2] = [clusters[k] + clusters[k + 1]]
                merged = True
                break
    out = [0.0] * len(values)
    for cluster in clusters:
        for i, pos in zip(cluster, layout(cluster)):
            out[i] = pos
    return out


def _direct_labels(ax, items: Sequence[Tuple[float, float, str]], dx_frac: float = 0.02,
                   dy_frac: float = 0.0, min_gap_frac: float = 0.065, group_frac: float = 0.25) -> None:
    """Label points to their right without overlaps.

    ``items`` are (x, y, text). Labels whose x positions are close (within
    ``group_frac`` of the x range) are spread vertically; a thin leader line is
    drawn when a label had to move away from its point.
    """
    if not items:
        return
    ax.relim()
    ax.autoscale_view()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr, yr = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    min_gap = min_gap_frac * yr

    # Single-linkage groups along x.
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    groups, current = [], [order[0]]
    for i in order[1:]:
        if items[i][0] - items[current[-1]][0] <= group_frac * xr:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    target_y = [0.0] * len(items)
    for group in groups:
        desired = [items[i][1] + dy_frac * yr for i in group]
        for i, pos in zip(group, _spread(desired, min_gap)):
            target_y[i] = pos

    for (x, y, text), ty in zip(items, target_y):
        moved = abs(ty - (y + dy_frac * yr)) > 0.3 * min_gap
        ax.annotate(
            text, xy=(x, y), xytext=(x + dx_frac * xr, ty), textcoords="data",
            fontsize=9, color=TEXT, ha="left", va="center", annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.6, shrinkA=0, shrinkB=3) if moved else None,
        )


def tradeoff_plot(summary: pd.DataFrame, y_metrics: Sequence[str] = ("vel_err", "rp_rate_rms")):
    """Small multiples sharing the x axis (cost of transport)."""
    fig, axes = plt.subplots(1, len(y_metrics), figsize=(5.2 * len(y_metrics), 4.2), sharex=True)
    if len(y_metrics) == 1:
        axes = [axes]
    for ax, ym in zip(axes, y_metrics):
        items = []
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
            items.append((r["cot"], r[ym], _label(st)))
        _style_axes(ax)
        ax.set_xlabel(METRIC_INFO["cot"][0] + "  (lower = more efficient)")
        ax.set_ylabel(METRIC_INFO[ym][0] + "  (lower = better)")
        # Slightly above the point, clear of the horizontal error bar.
        _direct_labels(ax, items, dx_frac=0.02, dy_frac=0.045)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Trade-offs between gait personalities (mean, 95% CI)", color=TEXT, fontsize=12)
    fig.tight_layout()
    plt.close(fig)
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
    plt.close(fig)
    return fig


def training_curves(histories: Dict[str, List[dict]]):
    """Evaluation reward vs. environment steps for each training run.

    Personalities are fine-tuned from the final standard policy, so their curves
    are drawn as a continuation (x = standard steps + fine-tuning steps).
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    base_hist = histories.get(BASE_STYLE) or []
    base_end = base_hist[-1]["step"] if base_hist else 0
    items, n_series, n_finetuned = [], 0, 0
    for name, hist in histories.items():
        if not hist:
            continue
        st = name.split("/")[-1]
        offset = 0 if st == BASE_STYLE else base_end
        n_series += 1
        n_finetuned += st != BASE_STYLE
        x = [(h["step"] + offset) / 1e6 for h in hist]
        y = [h["reward"] for h in hist]
        ax.plot(x, y, color=STYLE_COLORS.get(st, TEXT_2), linewidth=1.8,
                marker=STYLE_MARKERS.get(st, "o"), markersize=5, label=_label(st))
        items.append((x[-1], y[-1], _label(st), st == BASE_STYLE))
    _style_axes(ax)
    ax.set_xlabel("Environment steps [million]")
    ax.set_ylabel("Evaluation episode reward")
    if n_finetuned and base_end:
        # The fine-tuned curves start where the standard curve ends, so the boundary
        # is marked and the two sides are named instead of labelling the standard end point.
        x_b = base_end / 1e6
        ax.axvline(x_b, color=TEXT_2, linestyle=":", linewidth=1.0, alpha=0.7)
        for text, dx, ha in (("← " + _label(BASE_STYLE).lower(), -4, "right"), ("fine-tuning →", 4, "left")):
            ax.annotate(text, (x_b, 0.0), xycoords=("data", "axes fraction"), xytext=(dx, 4),
                        textcoords="offset points", ha=ha, va="bottom", fontsize=8, color=TEXT_2)
        ax.set_title("Each style is scored by its own reward, so values are not comparable across styles",
                     fontsize=8.5, color=TEXT_2)
        items = [it for it in items if not it[3]]
    _direct_labels(ax, [it[:3] for it in items], dx_frac=0.015, group_frac=0.03)
    if n_series >= 2:
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    plt.close(fig)
    return fig
