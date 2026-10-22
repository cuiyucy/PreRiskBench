from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
EXPORT_FORMATS = (".svg", ".pdf", ".png", ".tiff")
RASTER_DPI = 600

from data_utils import assert_source_totals, load_knn
from style import (
    MODE_COLORS,
    MODE_LABELS,
    MODE_ORDER,
    MUTED,
    TEXT,
    apply_style,
    clean_axis,
    luminance,
    panel_label,
    save_figure,
)


DIRECTION_ORDER = [
    "vitaldb->mover",
    "mover->vitaldb",
    "vitaldb->inspire",
    "inspire->vitaldb",
    "mover->inspire",
    "inspire->mover",
]
DIRECTION_LABELS = {
    "vitaldb->mover": "VitalDB → MOVER",
    "mover->vitaldb": "MOVER → VitalDB",
    "vitaldb->inspire": "VitalDB → INSPIRE",
    "inspire->vitaldb": "INSPIRE → VitalDB",
    "mover->inspire": "MOVER → INSPIRE",
    "inspire->mover": "INSPIRE → MOVER",
}


def build_figure() -> plt.Figure:
    apply_style(9.6, font_family="Times New Roman")
    assert_source_totals()
    df = load_knn()

    fig = plt.figure(figsize=(7.2, 4.1), constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.18, top=0.91, wspace=0.42)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.26, 1.04])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # a | Heatmap: zero-hit rate plus shared-feature count.
    zero = np.zeros((len(DIRECTION_ORDER), len(MODE_ORDER)))
    shared = np.zeros_like(zero, dtype=int)
    for i, direction in enumerate(DIRECTION_ORDER):
        for j, mode in enumerate(MODE_ORDER):
            row = df[(df["direction"] == direction) & (df["mode"] == mode)].iloc[0]
            zero[i, j] = float(row["zero_hit_rate"]) * 100
            shared[i, j] = int(row["total_shared_items"])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "zero_hit",
        ["#F7FBFF", "#CFE6F2", "#F1C6BF", "#B64342"],
    )
    norm = mpl.colors.Normalize(0, 100)
    image = ax_a.imshow(zero, cmap=cmap, norm=norm, aspect="auto")
    for i in range(zero.shape[0]):
        for j in range(zero.shape[1]):
            rgba = cmap(norm(zero[i, j]))
            colour = "white" if (0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]) < 0.48 else TEXT
            ax_a.text(j, i, f"{zero[i, j]:.1f}%\nn = {shared[i, j]}", ha="center", va="center", fontsize=9.2, color=colour, fontweight="bold")
    ax_a.set_xticks(np.arange(len(MODE_ORDER)), [MODE_LABELS[m].replace("-", "-\n") for m in MODE_ORDER])
    ax_a.set_yticks(np.arange(len(DIRECTION_ORDER)), [DIRECTION_LABELS[d] for d in DIRECTION_ORDER])
    ax_a.set_xlabel("")
    ax_a.set_title("Zero-hit rate and shared text features", loc="left")
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_a, orientation="horizontal", fraction=0.07, pad=0.13, aspect=28)
    cbar.set_label("Queries with zero shared features (%)")
    cbar.set_ticks([0, 25, 50, 75, 100])
    panel_label(ax_a, "a", x=-0.39, y=1.04)

    # b | Connected trajectories show that more shared features generally reduce zero hits.
    markers = {"unaligned": "o", "aligned": "s", "aligned_clustered": "D"}
    for direction in DIRECTION_ORDER:
        sub = df[df["direction"] == direction].set_index("mode").loc[MODE_ORDER]
        xx = sub["total_shared_items"].to_numpy(float)
        yy = sub["zero_hit_rate"].to_numpy(float) * 100
        xplot = np.log1p(xx)
        ax_b.plot(xplot, yy, color="#C7C7C7", lw=0.9, zorder=1)
        for mode, x_value, y_value in zip(MODE_ORDER, xplot, yy):
            ax_b.scatter(
                x_value,
                y_value,
                s=30,
                marker=markers[mode],
                facecolor=MODE_COLORS[mode],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
    tick_values = np.array([0, 10, 50, 250])
    ax_b.set_xticks(np.log1p(tick_values), ["0", "10", "50", "250"])
    ax_b.set_ylim(-2, 104)
    ax_b.set_xlabel("Shared text features (log1p scale)")
    ax_b.set_ylabel("Zero-hit rate (%)")
    ax_b.set_title("Coverage remains direction-dependent", loc="left")
    legend = [
        Line2D([0], [0], marker=markers[m], color="none", markerfacecolor=MODE_COLORS[m], markeredgecolor="white", markersize=6, label=MODE_LABELS[m])
        for m in MODE_ORDER
    ]
    ax_b.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.0, 0.16), fontsize=9.2, handletextpad=0.4)
    ax_b.text(0.02, 0.02, "Lines connect versions within one\nquery → reference direction", transform=ax_b.transAxes, fontsize=9.2, color=MUTED)
    clean_axis(ax_b)
    panel_label(ax_b, "b", x=-0.18, y=1.04)
    return fig


def main() -> None:
    outputs = save_figure(build_figure(), "fig04_retrieval_coverage")
    print("\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
