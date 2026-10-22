from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = FIGURE_ROOT / "outputs"

DATASET_ORDER = ["VitalDB", "MOVER", "INSPIRE"]
DATASET_COLORS = {
    "VitalDB": "#4C78A8",
    "MOVER": "#59A6A6",
    "INSPIRE": "#A277B5",
}

MODE_ORDER = ["unaligned", "aligned", "aligned_clustered"]
MODE_LABELS = {
    "unaligned": "Field-Aligned",
    "aligned": "Concept-Aligned",
    "aligned_clustered": "Concept-Clustered",
    "clustered": "Concept-Clustered",
}
MODE_COLORS = {
    "unaligned": "#B8B8B8",
    "aligned": "#8EC7E8",
    "aligned_clustered": "#2676A5",
    "clustered": "#2676A5",
}

SCORE_ORDER = ["High", "Middle", "Low"]
SCORE_COLORS = {
    "High": "#4C78A8",
    "Middle": "#F2C45E",
    "Low": "#D95F59",
}

PAIR_ORDER = [
    "vitaldb ↔ mover",
    "vitaldb ↔ inspire",
    "mover ↔ inspire",
]
PAIR_LABELS = {
    "vitaldb ↔ mover": "VitalDB–MOVER",
    "vitaldb ↔ inspire": "VitalDB–INSPIRE",
    "mover ↔ inspire": "MOVER–INSPIRE",
}
PAIR_COLORS = {
    "vitaldb ↔ mover": "#3B6FB6",
    "vitaldb ↔ inspire": "#8A5DA8",
    "mover ↔ inspire": "#2A9D8F",
}

TEXT = "#252525"
MUTED = "#555555"
LIGHT = "#E8E8E8"
NEGATIVE = "#B64342"
POSITIVE = "#2E8B57"


def apply_style(font_size: float = 7.5, font_family: str = "Arial") -> None:
    font_fallbacks = [font_family]
    if font_family == "Times New Roman":
        font_fallbacks.extend(["Times New Roman", "Times", "DejaVu Serif"])
    else:
        font_fallbacks.extend(["Arial", "Helvetica", "DejaVu Sans"])
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": font_fallbacks,
            "font.serif": font_fallbacks,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.titlesize": font_size + 0.5,
            "axes.labelsize": font_size,
            "axes.titleweight": "bold",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": font_size - 0.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.12,
    y: float = 1.04,
    font_size: float = 10,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=font_size,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=TEXT,
        clip_on=False,
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def save_figure(fig: plt.Figure, stem: str, dpi: int = 600) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    metadata = {"Creator": "Python/matplotlib reproducible figure pipeline"}
    for suffix in ("svg", "pdf", "png", "tiff"):
        path = OUTPUT_DIR / f"{stem}.{suffix}"
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
        }
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        if suffix == "pdf":
            kwargs["metadata"] = metadata
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def percent_axis(ax: plt.Axes, axis: str = "x") -> None:
    formatter = mpl.ticker.PercentFormatter(xmax=1.0, decimals=0)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)
