from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.textpath import TextPath

from style import DATASET_COLORS, MUTED, PROJECT_ROOT, TEXT, apply_style, save_figure

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

SCHEMA_FILE = PROJECT_ROOT / "data" / "analysis_source_schema_fields.csv"
OVERLAP_FILE = PROJECT_ROOT / "data" / "analysis_source_schema_overlap.csv"
RESOURCE_ORDER = ["VitalDB", "MOVER", "INSPIRE"]

# ---------------------------------------------------------------------------
# Geometry: all vertical measurements are in points measured from the top of
# the axes, converted to axes fractions by _y()/_dy(). This keeps every font
# pitch and gap at a fixed physical size when the figure height changes.
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 7.2, 3.40
_AX_H_PT = 0.985 * FIG_H * 72.0  # axes height in points (~205 pt)

CARD_W = 0.300
CARD_XS = (0.022, 0.350, 0.678)
CARD_CX = tuple(x + CARD_W / 2 for x in CARD_XS)
X0, X1 = 0.022, 0.978


def _y(pt: float) -> float:
    """points-from-top -> axes data y (top of axes is y = 1)."""
    return 1.0 - pt / _AX_H_PT


def _dy(pt: float) -> float:
    """positive point value -> axes-unit delta (downward)."""
    return pt / _AX_H_PT


# Panel a (source schemas)
A_TITLE_TOP = 3.0
A_CARD_TOP = 15.0
A_CARD_H = 58.0            # bottom at 73
ARROW_A = (76.0, 84.0)     # (top, bottom) in points-from-top

# Panel b (extraction and derivation)
B_TITLE_TOP = 87.0
B_CARD_TOP = 99.0
B_CARD_H = 72.0            # bottom at 171
ARROW_B = (174.0, 182.0)

# Panel c (common interface)
C_TITLE_TOP = 185.0
C_BAND_TOP = 197.0
C_BAND_H = 35.0            # bottom at 232

NOTE_COLOR = "#B85C38"
BORDER_SOFT = "#AAB6BF"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_inputs() -> tuple[dict[str, dict[str, str]], list[float]]:
    schema_rows = _load_rows(SCHEMA_FILE)
    overlap_rows = _load_rows(OVERLAP_FILE)
    by_resource = {row["resource"]: row for row in schema_rows}
    if set(RESOURCE_ORDER) - set(by_resource):
        raise ValueError("Source-schema audit is missing a required resource")
    overlaps = [float(row["jaccard_pct"]) for row in overlap_rows]
    if len(overlaps) != 3 or min(overlaps) != 3.51 or max(overlaps) != 7.32:
        raise ValueError(f"Unexpected schema-overlap values: {overlaps}")
    return by_resource, overlaps


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    lw: float = 0.9,
    rounding: float = 0.018,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            zorder=2,
        )
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9.5,
            linewidth=1.0,
            color="#87939C",
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def _shared_flow(ax: plt.Axes, top_pt: float, junction_pt: float, tip_pt: float) -> None:
    """Merge three cards before the next stage: every operation uses the sources."""
    for cx in CARD_CX:
        ax.plot([cx, cx], [_y(top_pt), _y(junction_pt)], color="#87939C", lw=1.0, zorder=3)
    ax.plot(
        [CARD_CX[0], CARD_CX[-1]],
        [_y(junction_pt), _y(junction_pt)],
        color="#87939C",
        lw=1.0,
        zorder=3,
    )
    _arrow(ax, (CARD_CX[1], _y(junction_pt)), (CARD_CX[1], _y(tip_pt)))


def _measure(text: str, fontsize: float) -> float:
    """Actual rendered width of a text string in points (Arial)."""
    return TextPath((0, 0), text, prop=FontProperties(family="Times New Roman", size=fontsize)).get_extents().width


def _wrap(text: str, fontsize: float, width_units: float, pad_pt: float = 16.0) -> list[str]:
    """Word-wrap by measured width so lines never overflow the box."""
    max_pt = width_units * FIG_W * 72.0 - pad_pt
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        if not cur or _measure(candidate, fontsize) <= max_pt:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = [text]
    # keep a trailing pipe attached to the following line for cleaner breaks
    out: list[str] = []
    pending = False
    for raw in lines:
        line = raw.strip()
        if pending:
            line = "| " + line
            pending = False
        if line.endswith("|"):
            line = line[:-1].rstrip()
            pending = True
        if line:
            out.append(line)
    if pending:
        out[-1] = out[-1] + " |"
    return out


def _block(
    ax: plt.Axes,
    x: float,
    y_top: float,
    lines: list[str],
    fontsize: float,
    color: str,
    *,
    bold: bool = False,
    pitch_pt: float = 7.0,
) -> float:
    """Draw wrapped lines stacked downward from y_top (va=top). Returns new cursor (axes y)."""
    pitch = _dy(pitch_pt)
    for line in lines:
        ax.text(
            x,
            y_top,
            line,
            fontsize=fontsize,
            color=color,
            fontweight="bold" if bold else "normal",
            ha="left",
            va="top",
        )
        y_top -= pitch
    return y_top


def _panel_header(
    ax: plt.Axes,
    label: str,
    title: str,
    top_pt: float,
    note: str | None = None,
    *,
    note_color: str = TEXT,
    note_bold: bool = False,
) -> None:
    y = _y(top_pt)
    ax.text(0.002, y, label, fontsize=10.0, fontweight="bold", color=TEXT, va="top")
    ax.text(0.032, y, title, fontsize=9.5, fontweight="bold", color=TEXT, va="top")
    if note:
        ax.text(
            X1,
            y,
            note,
            fontsize=8.6,
            color=note_color,
            fontweight="bold" if note_bold else "normal",
            ha="right",
            va="top",
        )


def _source_card(
    ax: plt.Axes,
    x: float,
    top_pt: float,
    h_pt: float,
    resource: str,
    table_label: str,
    keys_text: str,
    clinical_text: str,
) -> None:
    colour = DATASET_COLORS[resource]
    y_top = _y(top_pt)
    y_bot = _y(top_pt + h_pt)
    _box(ax, x, y_bot, CARD_W, y_top - y_bot, face=colour + "0D", edge=colour)
    pad = 0.018

    ax.text(x + pad, _y(top_pt + 3.8), resource, fontsize=9.0, fontweight="bold", color=colour, va="top")
    ax.plot(
        [x + pad, x + CARD_W - pad],
        [_y(top_pt + 13.5), _y(top_pt + 13.5)],
        color=colour,
        lw=0.7,
        alpha=0.5,
        solid_capstyle="round",
    )

    cur = _y(top_pt + 17.0)
    cur = _block(ax, x + pad, cur, _wrap(table_label, 8.6, CARD_W), 8.6, TEXT, pitch_pt=8.7)
    cur -= _dy(2.5)
    cur = _block(ax, x + pad, cur, _wrap(keys_text, 8.6, CARD_W), 8.6, TEXT, pitch_pt=8.7)
    cur -= _dy(2.5)
    _block(ax, x + pad, cur, _wrap(clinical_text, 8.6, CARD_W), 8.6, TEXT, pitch_pt=8.7)


def _process_card(
    ax: plt.Axes,
    x: float,
    top_pt: float,
    h_pt: float,
    title: str,
    steps: list[str],
    accent: str,
) -> None:
    y_top = _y(top_pt)
    y_bot = _y(top_pt + h_pt)
    _box(ax, x, y_bot, CARD_W, y_top - y_bot, face="#FAFBFD", edge=BORDER_SOFT)
    pad = 0.018

    ax.text(x + pad, _y(top_pt + 3.5), title, fontsize=8.7, fontweight="bold", color=TEXT, va="top")
    ax.plot(
        [x + pad, x + CARD_W - pad],
        [_y(top_pt + 13.5), _y(top_pt + 13.5)],
        color="#C4CDD4",
        lw=0.7,
        solid_capstyle="round",
    )

    cur = _y(top_pt + 17.0)
    for step in steps:
        lines = _wrap(step, 8.6, CARD_W - 0.028)
        ax.text(x + 0.020, cur + _dy(0.5), "•", fontsize=8.6, color=accent, va="top")
        cur = _block(ax, x + 0.036, cur, lines, 8.6, TEXT, pitch_pt=8.7)
        cur -= _dy(1.5)


def _output_band(
    ax: plt.Axes,
    x0: float,
    top_pt: float,
    w: float,
    h_pt: float,
    cells: list[tuple[str, list[str], str]],
) -> None:
    y_top = _y(top_pt)
    y_bot = _y(top_pt + h_pt)
    _box(ax, x0, y_bot, w, y_top - y_bot, face="#FBFCFD", edge=BORDER_SOFT)
    cellw = w / len(cells)
    for i, (title, detail, colour) in enumerate(cells):
        cx = x0 + (i + 0.5) * cellw
        ax.text(cx, _y(top_pt + 7.0), title, fontsize=8.7, fontweight="bold", color=colour, ha="center", va="center")
        ax.plot([cx - 0.048, cx + 0.048], [_y(top_pt + 12.0), _y(top_pt + 12.0)], color=colour, lw=0.8, solid_capstyle="round")
        if len(detail) == 1:
            ax.text(cx, _y(top_pt + 22.0), detail[0], fontsize=8.6, color=TEXT, ha="center", va="center")
        else:
            ax.text(cx, _y(top_pt + 19.0), detail[0], fontsize=8.6, color=TEXT, ha="center", va="center")
            ax.text(cx, _y(top_pt + 27.0), detail[1], fontsize=8.6, color=TEXT, ha="center", va="center")
        if i > 0:
            ax.plot([x0 + i * cellw, x0 + i * cellw], [y_bot + _dy(4.0), y_top - _dy(4.0)], color="#D6DDE3", lw=0.7)


def build_figure() -> plt.Figure:
    apply_style(8.8, font_family="Times New Roman")
    rows, overlaps = _validate_inputs()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.010, right=0.990, bottom=0.012, top=0.992)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    vital, mover, inspire = rows["VitalDB"], rows["MOVER"], rows["INSPIRE"]

    # ---------------- panel a: resource-specific source schemas ----------------
    _panel_header(
        ax,
        "a",
        "Resource-specific source schemas",
        A_TITLE_TOP,
        note=f" ",
    )
    sources = [
        (
            "VitalDB",
            "cases and resource-specific inputs",
            f"keys:  {vital['patient_identifier']}  |  {vital['surgery_identifier']}",
            f"clinical:  {vital['diagnosis']}  |  {vital['planned_procedure']}",
        ),
        (
            "MOVER",
            "epic_emr tables and repeated records",
            f"keys:  {mover['patient_identifier']}  |  {mover['surgery_identifier']}",
            "clinical:  diagnosis_code | procedure",
        ),
        (
            "INSPIRE",
            "operations and diagnosis tables",
            f"keys:  {inspire['patient_identifier']}  |  op_id / case_id",
            f"clinical:  {inspire['diagnosis']}  |  {inspire['planned_procedure']}",
        ),
    ]
    for x, (res, table, keys, clin) in zip(CARD_XS, sources):
        _source_card(ax, x, A_CARD_TOP, A_CARD_H, res, table, keys, clin)
    _shared_flow(ax, ARROW_A[0], 79.0, ARROW_A[1])

    # ---------------- panel b: extraction and derivation ----------------
    _panel_header(ax, "b", "Resource-specific extraction and derivation", B_TITLE_TOP)
    processes = [
        (
            "Link records and assign keys",
            [
                "Join source tables at surgery level",
                "Resolve repeated surgeries per patient",
                "Assign patient_id and surgery_id",
            ],
            DATASET_COLORS["VitalDB"],
        ),
        (
            "Derive structured variables",
            [
                "Query and aggregate long-form labs",
                "Apply preoperative window and validity rules",
                "Normalize names/units; pivot 27 labs",
            ],
            DATASET_COLORS["MOVER"],
        ),
        (
            "Derive clinical text",
            [
                "Clean diagnoses, procedures, and medications",
                "Resolve codes or source text",
                "Organize 10 histories + three text fields",
            ],
            DATASET_COLORS["INSPIRE"],
        ),
    ]
    for x, (title, steps, accent) in zip(CARD_XS, processes):
        _process_card(ax, x, B_CARD_TOP, B_CARD_H, title, steps, accent)
    _shared_flow(ax, ARROW_B[0], 177.0, ARROW_B[1])

    # ---------------- panel c: common benchmark interface ----------------
    _panel_header(ax, "c", "Common preoperative benchmark interface", C_TITLE_TOP)
    cells = [
        ("Common keys", ["patient_id · surgery_id"], "#4C78A8"),
        ("Structured inputs", ["demographics · anesthesia", "27 laboratory variables"], "#59A6A6"),
        ("Clinical text", ["Diagnosis · procedure", "Medication · 10 histories"], "#A277B5"),
        ("Fixed task label", ["ASA I–V", "(VitalDB I–IV)"], "#E58C45"),
    ]
    _output_band(ax, X0, C_BAND_TOP, X1 - X0, C_BAND_H, cells)

    return fig


def main() -> None:
    outputs = save_figure(build_figure(), "fig01_field_alignment")
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
