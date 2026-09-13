'''
Schematic figure for the two-stage graph-based merge procedure (Algorithm 1).

Shows fine cells with transition edges (tau_tr, blue arrows) and value-space
edges (tau_val, orange dashed lines between cluster centroids) in one panel.

Usage:
  python generate_merge_diagram.py
  python generate_merge_diagram.py --show
'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from matplotlib.path import Path as MplPath

_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = _ROOT / "docs" / "figures" / "algorithm" / "merge_graph_schematic.png"

# Layout and typography — edit before regenerating
FIGSIZE = (7.0, 4.2)
DPI = 150
FONT_SIZE = 11
NODE_RADIUS = 0.28
AXIS_XLIM = (-0.3, 9.2)
AXIS_YLIM = (-0.3, 3.8)

# Fine cell centers (x, y)
FINE_CELL_POS = {
    "alpha": (0.9, 2.1),
    "beta": (2.1, 2.5),
    "gamma": (1.5, 0.85),
    "delta": (3.9, 2.3),
    "epsilon": (4.8, 0.95),
    "zeta": (7.8, 2.1),
    "eta_discarded": (0.15, 0.15),
}

# Undirected transition edges when max(p_ab, p_ba) >= tau_tr
TRANSITION_EDGES = [
    ("alpha", "beta"),
    ("beta", "gamma"),
    ("gamma", "alpha"),
    ("delta", "epsilon"),
]

# Stage-1 connected components (ovals)
STAGE1_GROUPS = [
    {"cells": ["alpha", "beta", "gamma"], "color": "#dbeafe"},
    {"cells": ["delta", "epsilon"], "color": "#bbf7d0"},
]

# Final merged components (ovals)
FINAL_GROUPS = [
    {"label": r"$D_1$", "cells": ["alpha", "beta", "gamma", "delta", "epsilon"], "color": "#c5cad3"},
    {"label": r"$D_2$", "cells": ["zeta"], "color": "#9ca3af"},
]

OVAL_PAD = 0.42
OVAL_ALPHA = 0.55
D1_OVAL_EXTRA_PAD = 0.28  # margin beyond stage-1 ovals
D2_OVAL_PAD = OVAL_PAD + 0.18

# Value-space edge between centroids of two stage-1 groups (tau_val)
VALUE_EDGE_GROUPS = (
    ["alpha", "beta", "gamma"],
    ["delta", "epsilon"],
)


def _finalize_figure(fig, *, save_path: Path | None, show: bool) -> None:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor="white")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _group_centroid(cell_ids: list[str], positions: dict[str, tuple[float, float]]) -> tuple[float, float]:
    pts = np.asarray([positions[cid] for cid in cell_ids], dtype=float)
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1])


def _group_ellipse(
    cell_ids: list[str],
    positions: dict[str, tuple[float, float]],
    color: str,
    *,
    pad: float = OVAL_PAD,
    alpha: float = OVAL_ALPHA,
    zorder: int = 0,
) -> Ellipse:
    pts = np.asarray([positions[cid] for cid in cell_ids], dtype=float)
    cx, cy = pts.mean(axis=0)
    half_w = max((pts[:, 0].max() - pts[:, 0].min()) / 2.0 + NODE_RADIUS + pad, NODE_RADIUS + pad)
    half_h = max((pts[:, 1].max() - pts[:, 1].min()) / 2.0 + NODE_RADIUS + pad, NODE_RADIUS + pad)
    return Ellipse(
        (cx, cy),
        width=2.0 * half_w,
        height=2.0 * half_h,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
    )


def _ellipse_bounds(
    cell_ids: list[str],
    positions: dict[str, tuple[float, float]],
    *,
    pad: float,
) -> tuple[float, float, float, float]:
    ellipse = _group_ellipse(cell_ids, positions, "#000000", pad=pad)
    cx, cy = ellipse.center
    half_w = ellipse.width / 2.0
    half_h = ellipse.height / 2.0
    return cx - half_w, cx + half_w, cy - half_h, cy + half_h


def _enclosing_ellipse(
    bounds_list: list[tuple[float, float, float, float]],
    color: str,
    *,
    extra_pad: float = 0.0,
    alpha: float = OVAL_ALPHA,
    zorder: int = 0,
) -> Ellipse:
    x_min = min(b[0] for b in bounds_list) - extra_pad
    x_max = max(b[1] for b in bounds_list) + extra_pad
    y_min = min(b[2] for b in bounds_list) - extra_pad
    y_max = max(b[3] for b in bounds_list) + extra_pad
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    return Ellipse(
        (cx, cy),
        width=x_max - x_min,
        height=y_max - y_min,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
    )


def _draw_d1_oval(ax, group: dict, positions: dict[str, tuple[float, float]]) -> None:
    stage_bounds = [
        _ellipse_bounds(cells, positions, pad=OVAL_PAD)
        for cells in (g["cells"] for g in STAGE1_GROUPS)
    ]
    ax.add_patch(
        _enclosing_ellipse(
            stage_bounds,
            group["color"],
            extra_pad=D1_OVAL_EXTRA_PAD,
            zorder=0,
        )
    )


def draw_fine_cell(
    ax,
    center: tuple[float, float],
    label: str,
    *,
    facecolor: str = "white",
    edgecolor: str = "#374151",
    alpha: float = 1.0,
    linestyle: str = "solid",
    zorder: int = 3,
) -> Circle:
    circle = Circle(
        center,
        NODE_RADIUS,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.4,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(circle)
    ax.text(
        center[0],
        center[1],
        label,
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        zorder=zorder + 1,
    )
    return circle


def draw_transition_edge(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    color: str = "#1d4ed8",
    zorder: int = 2,
) -> None:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return
    unit = direction / norm
    start = p0 + unit * NODE_RADIUS * 0.85
    end = p1 - unit * NODE_RADIUS * 0.85
    mid = 0.5 * (start + end)
    perp = np.array([-unit[1], unit[0]])
    rad = 0.35 * norm
    ctrl = mid + perp * rad

    path = MplPath(
        [start, ctrl, end],
        [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
    )
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.5,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_value_edge(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    color: str = "#ea580c",
    zorder: int = 2,
) -> None:
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        linestyle="--",
        linewidth=2.0,
        color=color,
        zorder=zorder,
    )


def draw_centroid_marker(
    ax,
    center: tuple[float, float],
    label: str,
    *,
    color: str = "#ea580c",
    label_offset: tuple[float, float] = (0.0, -0.22),
    zorder: int = 4,
) -> None:
    ax.plot(
        center[0],
        center[1],
        marker="o",
        markersize=7,
        color="#111827",
        markeredgecolor=color,
        markeredgewidth=1.2,
        zorder=zorder,
    )
    ax.text(
        center[0] + label_offset[0],
        center[1] + label_offset[1],
        label,
        ha="center",
        va="top",
        fontsize=FONT_SIZE - 1,
        color="#111827",
        zorder=zorder + 1,
    )


def draw_schematic(ax) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(*AXIS_XLIM)
    ax.set_ylim(*AXIS_YLIM)

    # Back to front: final components, then stage-1 clusters
    for group in FINAL_GROUPS:
        if group["label"] == r"$D_1$":
            _draw_d1_oval(ax, group, FINE_CELL_POS)
        else:
            ax.add_patch(
                _group_ellipse(
                    group["cells"],
                    FINE_CELL_POS,
                    group["color"],
                    pad=D2_OVAL_PAD,
                    zorder=0,
                )
            )

    for group in STAGE1_GROUPS:
        ax.add_patch(
            _group_ellipse(group["cells"], FINE_CELL_POS, group["color"], pad=OVAL_PAD, zorder=1)
        )

    for group in FINAL_GROUPS:
        if group["label"] == r"$D_2$":
            cx, cy = FINE_CELL_POS["zeta"]
            ax.text(
                cx,
                cy + NODE_RADIUS + 0.55,
                group["label"],
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE,
                fontweight="bold",
            )
        else:
            pts = np.asarray([FINE_CELL_POS[c] for c in group["cells"]], dtype=float)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].max() + NODE_RADIUS + 0.75)
            ax.text(
                cx,
                cy,
                group["label"],
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE,
                fontweight="bold",
            )

    group_a, group_b = VALUE_EDGE_GROUPS
    m1 = _group_centroid(group_a, FINE_CELL_POS)
    m2 = _group_centroid(group_b, FINE_CELL_POS)
    draw_value_edge(ax, m1, m2)
    draw_centroid_marker(ax, m1, r"$m_1$", label_offset=(-0.05, -0.24))
    draw_centroid_marker(ax, m2, r"$m_2$", label_offset=(0.05, -0.24))

    for a, b in TRANSITION_EDGES:
        draw_transition_edge(ax, FINE_CELL_POS[a], FINE_CELL_POS[b])

    labels = {
        "alpha": r"$C_\alpha$",
        "beta": r"$C_\beta$",
        "gamma": r"$C_\gamma$",
        "delta": r"$C_\delta$",
        "epsilon": r"$C_\varepsilon$",
        "zeta": r"$C_\zeta$",
    }
    for cid, text in labels.items():
        draw_fine_cell(ax, FINE_CELL_POS[cid], text)

    draw_fine_cell(
        ax,
        FINE_CELL_POS["eta_discarded"],
        r"$w_\alpha<\tau_{\mathrm{occ}}$",
        facecolor="#f9fafb",
        edgecolor="#9ca3af",
        alpha=0.55,
        linestyle="dashed",
    )


def generate_merge_diagram(*, output_path: Path = OUTPUT_PATH, show: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    draw_schematic(ax)
    fig.tight_layout()
    _finalize_figure(fig, save_path=output_path, show=show)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate merge algorithm schematic figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output PNG path.",
    )
    parser.add_argument("--show", action="store_true", help="Display interactively.")
    args = parser.parse_args()
    generate_merge_diagram(output_path=args.output.resolve(), show=args.show)


if __name__ == "__main__":
    main()
