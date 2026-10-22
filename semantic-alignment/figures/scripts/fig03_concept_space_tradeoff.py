from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
EXPORT_FORMATS = (".svg", ".pdf", ".png", ".tiff")
RASTER_DPI = 600

from data_utils import assert_source_totals, load_concept_space
from style import (
    MODE_LABELS,
    MODE_ORDER,
    PAIR_COLORS,
    PAIR_LABELS,
    PAIR_ORDER,
    apply_style,
    clean_axis,
    panel_label,
    save_figure,
)


def build_figure() -> plt.Figure:
    apply_style(9.6, font_family="Times New Roman")
    assert_source_totals()
    concept = load_concept_space()

    fig = plt.figure(figsize=(7.2, 2.9), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.975, bottom=0.21, top=0.88, wspace=0.42)
    gs = fig.add_gridspec(1, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    pair_rows = concept[concept["pair"].isin(PAIR_ORDER)].copy()
    x = np.arange(len(MODE_ORDER))

    # a | Pairwise Jaccard overlap.
    for pair in PAIR_ORDER:
        sub = pair_rows[pair_rows["pair"] == pair].set_index("mode").loc[MODE_ORDER]
        values = sub["jaccard_pct"].to_numpy(float)
        ax_a.plot(x, values, marker="o", ms=5.0, lw=1.8, color=PAIR_COLORS[pair], label=PAIR_LABELS[pair])
        ax_a.text(x[-1] + 0.06, values[-1], f"{values[-1]:.2f}%", va="center", fontsize=8.6, color=PAIR_COLORS[pair])
    ax_a.set_xticks(x, [MODE_LABELS[m] for m in MODE_ORDER])
    ax_a.set_ylabel("Pairwise Jaccard overlap (%)")
    ax_a.set_ylim(-0.4, 15.5)
    ax_a.set_xlim(-0.08, 2.45)
    ax_a.set_title("Pairwise overlap increases", loc="left")
    ax_a.legend(loc="upper left", ncol=1)
    clean_axis(ax_a)
    panel_label(ax_a, "a")

    # b | Jensen-Shannon divergence.
    for pair in PAIR_ORDER:
        sub = pair_rows[pair_rows["pair"] == pair].set_index("mode").loc[MODE_ORDER]
        values = sub["js_divergence"].to_numpy(float)
        ax_b.plot(x, values, marker="o", ms=5.0, lw=1.8, color=PAIR_COLORS[pair])
        offset = {"vitaldb ↔ mover": 0.010, "vitaldb ↔ inspire": 0.0, "mover ↔ inspire": -0.010}[pair]
        ax_b.text(x[-1] + 0.06, values[-1] + offset, f"{values[-1]:.3f}", va="center", fontsize=8.6, color=PAIR_COLORS[pair])
    ax_b.set_xticks(x, [MODE_LABELS[m] for m in MODE_ORDER])
    ax_b.set_ylabel("Jensen–Shannon divergence")
    ax_b.set_ylim(0.47, 0.95)
    ax_b.set_xlim(-0.08, 2.45)
    ax_b.set_title("Residual distributions remain distinct", loc="left")
    clean_axis(ax_b)
    panel_label(ax_b, "b", x=-0.15)

    return fig


def main() -> None:
    outputs = save_figure(build_figure(), "fig03_concept_space_tradeoff")
    print("\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
