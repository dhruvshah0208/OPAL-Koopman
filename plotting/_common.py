"""Shared bits for the plotting scripts: locate cached arrays, save/show figures.

Each script reads one ``data/paper_plots/<system>/<name>.npz`` written by
``notebooks/paper_plots.ipynb`` and draws the matching figure, so figure styling can be
edited here without re-running the notebook. Run from the repository root, e.g.

    python plotting/plot_partition.py duffing
    python plotting/plot_rollouts.py bistable_oscillations --show
"""

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/paper_plots"
FIG_DIR = Path(__file__).resolve().parent   # figures are written next to the scripts as <system>_<name>.png

# Shared styling
SWITCHED_LABEL = "regional EDMD"
VANILLA_LABEL = "vanilla EDMD"
COLOR_TRUE, COLOR_SWITCHED, COLOR_VANILLA = "black", "crimson", "#1f4fd8"
LW_TRUE, LW_SWITCHED, LW_VANILLA = 2.6, 2.2, 2.2
CLASS_COLORS = ["red", "blue", "green", "orange"]          # partition cell / class colors
CELL_FILL_COLORS = ["#f7b66b", "#7fcf8a", "#b89ae0", "#f090ad"]  # same hues as before, more saturated so the cells read clearly
CELL_POINT_ALPHA, CELL_POINT_SIZE = 0.8, 3
DPI = 150

# Font sizes (points); applied to every script through matplotlib's rcParams on import.
FONT_AXIS_LABEL, FONT_TICK, FONT_LEGEND = 16, 13, 13
import matplotlib as _mpl  # noqa: E402

_mpl.rcParams.update({
    "axes.labelsize": FONT_AXIS_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
})


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("system", help="duffing or bistable_oscillations (subfolder of data/paper_plots)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"default: {DATA_DIR}")
    parser.add_argument("--out", type=Path, default=FIG_DIR, help=f"output directory (default: {FIG_DIR})")
    parser.add_argument("--show", action="store_true", help="also open the figure window")
    return parser.parse_args()


def load(args: argparse.Namespace, name: str) -> dict:
    path = args.data_dir / args.system / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run notebooks/paper_plots.ipynb first.")
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    if "system_name" in out:
        out["system_name"] = str(out["system_name"])
    return out


def finish(fig, args: argparse.Namespace, name: str) -> None:
    """Save ``fig`` as <out>/<system>_<name>.png (and show it if requested)."""
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.system}_{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Plot saved to {path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)
