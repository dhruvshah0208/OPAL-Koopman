'''
Regenerate paper figures for switched local Koopman models from saved plot_data .npz.

Reads arrays written by build_multiple_linear_models.save_plot_data and writes:
  docs/figures/<system>/multiple_linear_models/partition_scatter.png
  docs/figures/<system>/multiple_linear_models/phase_trajectories.png

Usage:
  python scripts/generate_plots.py
  python scripts/generate_plots.py configs/duffing/build_multiple_linear_models.json
  python scripts/generate_plots.py data/duffing/multiple_linear_models_plot_data.npz
'''

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from omega_partition.paths import DOCS_FIGURES_DIR, PROJECT_ROOT, resolve  # noqa: E402
from omega_partition.plotting import finalize_figure  # noqa: E402

_ROOT = PROJECT_ROOT
_DOCS_FIGURES = DOCS_FIGURES_DIR

DEFAULT_PLOT_CONFIGS = (
    _ROOT / "configs/duffing/build_multiple_linear_models.json",
    _ROOT / "configs/bistable_oscillations/build_multiple_linear_models.json",
)

# Paper figure typography — edit these before running generate_plots.py
PAPER_SCATTER_LABEL_FONT_SIZE = 16
PAPER_SCATTER_TICK_FONT_SIZE = 14
PAPER_LABEL_FONT_SIZE = 14
PAPER_TICK_FONT_SIZE = 12
PAPER_LEGEND_FONT_SIZE = 12

from build_multiple_linear_models import (  # noqa: E402
    _settings,
    load_plot_data,
    rollouts_from_arrays,
)


def _apply_paper_axis_fonts(
    ax,
    *,
    label_font_size: float,
    tick_font_size: float,
    legend_font_size: float | None = None,
) -> None:
    ax.xaxis.label.set_fontsize(label_font_size)
    ax.yaxis.label.set_fontsize(label_font_size)
    ax.tick_params(axis="both", which="both", labelsize=tick_font_size)
    legend = ax.get_legend()
    if legend is not None and legend_font_size is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_font_size)


def _plot_state_partition_for_paper(
    X,
    labels,
    *,
    n_state: int,
    save_path: Path | None = None,
    show: bool = False,
    label_font_size: float = PAPER_SCATTER_LABEL_FONT_SIZE,
    tick_font_size: float = PAPER_SCATTER_TICK_FONT_SIZE,
):
    """Paper-formatted partition scatter: no title, no colorbar."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = np.asarray(X)
    labels = np.asarray(labels).ravel()
    partition_cmap = ListedColormap(["red", "blue", "green", "orange"])
    partition_cmap.set_under("lightgray")

    if n_state == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=labels,
            s=4,
            alpha=0.45,
            edgecolors="none",
            cmap=partition_cmap,
            vmin=0,
            vmax=len(partition_cmap.colors) - 1,
        )
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    elif n_state == 1:
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.05, 0.05, size=X.shape[0])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.scatter(
            X[:, 0],
            jitter,
            c=labels,
            s=4,
            alpha=0.45,
            edgecolors="none",
            cmap=partition_cmap,
            vmin=0,
            vmax=len(partition_cmap.colors) - 1,
        )
        ax.set_xlabel(r"$x_1$")
        ax.set_yticks([])
    else:
        raise ValueError(f"Unsupported state dimension n={n_state} for partition scatter.")

    ax.grid(alpha=0.3)
    _apply_paper_axis_fonts(
        ax,
        label_font_size=label_font_size,
        tick_font_size=tick_font_size,
    )
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def _plot_phase_trajectories_for_paper(
    rollouts: list,
    *,
    save_path: Path | None = None,
    show: bool = False,
    label_font_size: float = PAPER_LABEL_FONT_SIZE,
    tick_font_size: float = PAPER_TICK_FONT_SIZE,
    legend_font_size: float = PAPER_LEGEND_FONT_SIZE,
):
    """Paper-formatted phase trajectories: no figure title or subplot titles."""
    import matplotlib.pyplot as plt

    if not rollouts:
        return None

    n_state = rollouts[0][0].shape[1]
    horizon = rollouts[0][0].shape[0]
    fig, axes = plt.subplots(n_state, 1, figsize=(8, 3.0 * n_state), squeeze=False)
    t = np.arange(horizon)

    for j in range(n_state):
        ax = axes[j, 0]
        for i, (true_traj, pred_traj, _) in enumerate(rollouts):
            valid = np.isfinite(pred_traj[:, j])
            if not np.any(valid):
                continue
            t_valid = t[valid]
            ax.plot(
                t_valid,
                true_traj[valid, j],
                color="black",
                lw=1.2,
                alpha=0.65,
                label="true" if i == 0 else None,
            )
            ax.plot(
                t_valid,
                pred_traj[valid, j],
                color="crimson",
                lw=1.2,
                ls="--",
                alpha=0.85,
                label="EDMD" if i == 0 else None,
            )
        ax.set_ylabel(rf"$x_{j + 1}$")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        _apply_paper_axis_fonts(
            ax,
            label_font_size=label_font_size,
            tick_font_size=tick_font_size,
            legend_font_size=legend_font_size,
        )

    axes[-1, 0].set_xlabel("time step")
    _apply_paper_axis_fonts(
        axes[-1, 0],
        label_font_size=label_font_size,
        tick_font_size=tick_font_size,
    )
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def _plot_data_path_from_config(config_path: Path) -> Path:
    cfg = _settings(config_path)
    return resolve(cfg["plot_data_output_path"])


def generate_paper_plots(
    plot_data_path: Path | str,
    *,
    output_dir: Path | None = None,
    show: bool = False,
    scatter_label_font_size: float = PAPER_SCATTER_LABEL_FONT_SIZE,
    scatter_tick_font_size: float = PAPER_SCATTER_TICK_FONT_SIZE,
    label_font_size: float = PAPER_LABEL_FONT_SIZE,
    tick_font_size: float = PAPER_TICK_FONT_SIZE,
    legend_font_size: float = PAPER_LEGEND_FONT_SIZE,
) -> Path:
    """Render partition scatter and phase trajectory figures used in the paper."""
    data = load_plot_data(plot_data_path)
    out_dir = output_dir or (_DOCS_FIGURES / data["docs_figures_relpath"])
    out_dir.mkdir(parents=True, exist_ok=True)

    X = data["partition_X"]
    labels = data["partition_labels"]
    n_state = data["n_state"]

    if n_state in (1, 2):
        _plot_state_partition_for_paper(
            X,
            labels,
            n_state=n_state,
            save_path=out_dir / "partition_scatter.png",
            show=show,
            label_font_size=scatter_label_font_size,
            tick_font_size=scatter_tick_font_size,
        )
    else:
        raise ValueError(
            f"Unsupported state dimension n={n_state} in {data['source_path']}. "
            "Only 1D and 2D partition scatter plots are implemented."
        )

    rollouts = rollouts_from_arrays(data["phase_true_trajs"], data["phase_pred_trajs"])
    if rollouts:
        _plot_phase_trajectories_for_paper(
            rollouts,
            save_path=out_dir / "phase_trajectories.png",
            show=show,
            label_font_size=label_font_size,
            tick_font_size=tick_font_size,
            legend_font_size=legend_font_size,
        )
    else:
        print(f"No phase trajectories stored in {data['source_path']}; skipped phase_trajectories.png")

    print(f"Paper figures written to {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        raise SystemExit(
            "usage: generate_plots.py [CONFIG.json | plot_data.npz]\n"
            "If omitted, regenerates figures for all default switched-model configs."
        )

    if len(argv) == 0:
        targets = [(_plot_data_path_from_config(path), path) for path in DEFAULT_PLOT_CONFIGS]
    else:
        arg = Path(argv[0]).resolve()
        if not arg.is_file():
            raise SystemExit(f"Input not found: {arg}")
        if arg.suffix == ".json":
            targets = [(_plot_data_path_from_config(arg), arg)]
        elif arg.suffix == ".npz":
            targets = [(arg, arg)]
        else:
            raise SystemExit(f"Unsupported input type: {arg}")

    for plot_data_path, label in targets:
        if not plot_data_path.is_file():
            raise FileNotFoundError(
                f"Plot data not found: {plot_data_path}\n"
                f"Run build_multiple_linear_models.py on {label} first."
            )
        print(f"Generating paper figures from {plot_data_path}")
        generate_paper_plots(plot_data_path, show=False)


if __name__ == "__main__":
    main()
