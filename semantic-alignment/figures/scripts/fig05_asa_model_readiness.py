from __future__ import annotations

import shutil

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
EXPORT_FORMATS = (".svg", ".pdf", ".png", ".tiff")
RASTER_DPI = 600

from data_utils import assert_source_totals, load_joint_clinicalbert
from style import MODE_COLORS, MODE_LABELS, PROJECT_ROOT, apply_style, clean_axis, panel_label, save_figure


ALGORITHM_ORDER = ["lr", "rf", "xgboost", "mlp", "cnn", "tabtransformer"]
ALGORITHM_LABELS = {
    "lr": "LR",
    "rf": "RF",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "cnn": "1D-CNN",
    "tabtransformer": "TabTransformer",
}
MODE_ORDER_LOCAL = ["unaligned", "aligned", "clustered"]


def build_figure() -> plt.Figure:
    # The manuscript places this figure across both columns.  A side-by-side
    # layout restores the intended wide aspect while keeping classifier labels
    # horizontal and all source text large enough for an 8-pt on-page floor.
    apply_style(10.5, font_family="Times New Roman")
    assert_source_totals()
    df = load_joint_clinicalbert()

    fig = plt.figure(figsize=(7.25, 3.45), constrained_layout=False)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.20, top=0.80, wspace=0.35)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

    y = np.arange(len(ALGORITHM_ORDER))
    shifts = {"unaligned": -0.18, "aligned": 0.0, "clustered": 0.18}
    markers = {"unaligned": "o", "aligned": "s", "clustered": "D"}
    for mode in MODE_ORDER_LOCAL:
        sub = df[df["mode"] == mode].set_index("algorithm").loc[ALGORITHM_ORDER]
        ax_a.errorbar(
            sub["macro_f1_mean"],
            y + shifts[mode],
            xerr=sub["macro_f1_sd"],
            fmt=markers[mode],
            color=MODE_COLORS[mode],
            markeredgecolor="white",
            markeredgewidth=0.65,
            markersize=5.8,
            linewidth=0,
            elinewidth=1.05,
            capsize=2.4,
            label=MODE_LABELS[mode],
        )
    ax_a.set_yticks(y, [ALGORITHM_LABELS[a] for a in ALGORITHM_ORDER])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.20, 0.515)
    ax_a.set_xlabel("Joint-training Macro-F1")
    ax_a.set_title("Absolute performance", loc="left", pad=7)
    ax_a.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    clean_axis(ax_a)
    panel_label(ax_a, "a", x=-0.24, y=1.07, font_size=11.5)

    field = df[df["mode"] == "unaligned"].set_index("algorithm").loc[ALGORITHM_ORDER]
    aligned = df[df["mode"] == "aligned"].set_index("algorithm").loc[ALGORITHM_ORDER]
    clustered = df[df["mode"] == "clustered"].set_index("algorithm").loc[ALGORITHM_ORDER]
    aligned_delta = aligned["macro_f1_mean"].to_numpy() - field["macro_f1_mean"].to_numpy()
    clustered_delta = clustered["macro_f1_mean"].to_numpy() - field["macro_f1_mean"].to_numpy()
    for values, mode, offset in [
        (aligned_delta, "aligned", -0.11),
        (clustered_delta, "clustered", 0.11),
    ]:
        ax_b.hlines(y + offset, 0, values, color=MODE_COLORS[mode], linewidth=1.35, alpha=0.88)
        ax_b.scatter(
            values,
            y + offset,
            s=31,
            marker=markers[mode],
            color=MODE_COLORS[mode],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
    ax_b.axvline(0, color="#6B6B6B", linewidth=1.0)
    ax_b.set_xlim(-0.0015, 0.0235)
    ax_b.set_xlabel("Change from Field-Aligned")
    ax_b.set_title("Within-classifier change", loc="left", pad=7)
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    ax_b.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    clean_axis(ax_b)
    panel_label(ax_b, "b", x=-0.18, y=1.07, font_size=11.5)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.985),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.35,
        fontsize=10.3,
    )
    return fig


def main() -> None:
    outputs = save_figure(build_figure(), "fig05_asa_model_readiness")
    manuscript_dir = PROJECT_ROOT / "paper" / "figures"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    pdf_output = next(path for path in outputs if path.suffix == ".pdf")
    shutil.copy2(pdf_output, manuscript_dir / pdf_output.name)
    print("\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
