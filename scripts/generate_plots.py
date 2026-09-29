'''
Regenerate paper figures for regional EDMD models from saved plot_data .npz.

Reads arrays written by build_multiple_linear_models.save_plot_data and writes:
  figures/paper/<system>/multiple_linear_models/partition_scatter.png
  figures/paper/<system>/multiple_linear_models/phase_trajectories.png
  figures/paper/<system>/multiple_linear_models/rollout_errors.png   (if rollout curves were saved)

When the plot data contains a vanilla-EDMD comparison, phase_trajectories.png
overlays it and rollout_errors.png shows both models.

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

from opal.paths import PAPER_FIGURES_DIR, PROJECT_ROOT, resolve  # noqa: E402
from opal.plotting import finalize_figure  # noqa: E402

_ROOT = PROJECT_ROOT
_PAPER_FIGURES = PAPER_FIGURES_DIR

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
    REGIONAL_LABEL,
    VANILLA_LABEL,
    load_settings,
    load_plot_data,
    rollouts_from_arrays,
    true_trajectory_ylim,
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
    vanilla_rollouts: list | None = None,
    save_path: Path | None = None,
    show: bool = False,
    label_font_size: float = PAPER_LABEL_FONT_SIZE,
    tick_font_size: float = PAPER_TICK_FONT_SIZE,
    legend_font_size: float = PAPER_LEGEND_FONT_SIZE,
):
    """Paper-formatted phase trajectories: no figure title or subplot titles.

    ``vanilla_rollouts`` (same true trajectories, vanilla-EDMD predictions) is
    overlaid when given.
    """
    import matplotlib.pyplot as plt

    if not rollouts:
        return None

    n_state = rollouts[0][0].shape[1]
    horizon = rollouts[0][0].shape[0]
    fig, axes = plt.subplots(n_state, 1, figsize=(8, 3.0 * n_state), squeeze=False)
    t = np.arange(horizon)

    # (rollout list, color, linestyle, alpha, legend label, tuple index of the trajectory)
    series = [
        (rollouts, "black", "-", 0.65, "true", 0),
        (rollouts, "crimson", "--", 0.85, REGIONAL_LABEL, 1),
    ]
    if vanilla_rollouts:
        series.append((vanilla_rollouts, "steelblue", ":", 0.9, VANILLA_LABEL, 1))

    for j in range(n_state):
        ax = axes[j, 0]
        for rolls, color, ls, alpha, label, which in series:
            for i, item in enumerate(rolls):
                traj = item[which][:, j]
                valid = np.isfinite(traj)
                if not np.any(valid):
                    continue
                ax.plot(
                    t[valid],
                    traj[valid],
                    color=color,
                    lw=1.2,
                    ls=ls,
                    alpha=alpha,
                    label=label if i == 0 else None,
                )
        ax.set_ylim(*true_trajectory_ylim(rollouts, j))
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


def _plot_rollout_errors_for_paper(
    rel_regional: np.ndarray,
    rel_vanilla: np.ndarray | None = None,
    *,
    save_path: Path | None = None,
    show: bool = False,
    label_font_size: float = PAPER_LABEL_FONT_SIZE,
    tick_font_size: float = PAPER_TICK_FONT_SIZE,
    legend_font_size: float = PAPER_LEGEND_FONT_SIZE,
):
    """Paper-formatted mean relative rollout error vs time step (log scale), with
    the per-trajectory curves drawn faintly behind the mean."""
    import matplotlib.pyplot as plt

    rel_regional = np.asarray(rel_regional, dtype=np.float64)
    if rel_regional.size == 0:
        return None

    curves = [(rel_regional, "crimson", REGIONAL_LABEL)]
    if rel_vanilla is not None and np.asarray(rel_vanilla).size > 0:
        curves.append((np.asarray(rel_vanilla, dtype=np.float64), "steelblue", VANILLA_LABEL))

    fig, ax = plt.subplots(figsize=(7, 4))
    for arr, color, label in curves:
        t = np.arange(arr.shape[1])
        for row in arr:
            ax.plot(t, row, color=color, lw=0.5, alpha=0.12)
        with np.errstate(all="ignore"):
            mean = np.nanmean(arr, axis=0)
        ax.plot(t, mean, color=color, lw=2.0, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$\|\hat{x}_t - x_t\| / \|x_t\|$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    _apply_paper_axis_fonts(
        ax,
        label_font_size=label_font_size,
        tick_font_size=tick_font_size,
        legend_font_size=legend_font_size,
    )
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def _plot_data_path_from_config(config_path: Path) -> Path:
    cfg = load_settings(config_path)
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
    out_dir = output_dir or (_PAPER_FIGURES / data["paper_figures_relpath"])
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
    vanilla_rollouts = rollouts_from_arrays(data["phase_true_trajs"], data["phase_vanilla_trajs"])
    if rollouts:
        _plot_phase_trajectories_for_paper(
            rollouts,
            vanilla_rollouts=vanilla_rollouts or None,
            save_path=out_dir / "phase_trajectories.png",
            show=show,
            label_font_size=label_font_size,
            tick_font_size=tick_font_size,
            legend_font_size=legend_font_size,
        )
    else:
        print(f"No phase trajectories stored in {data['source_path']}; skipped phase_trajectories.png")

    if data["rollout_rel_err_regional"].size > 0:
        _plot_rollout_errors_for_paper(
            data["rollout_rel_err_regional"],
            data["rollout_rel_err_vanilla"],
            save_path=out_dir / "rollout_errors.png",
            show=show,
            label_font_size=label_font_size,
            tick_font_size=tick_font_size,
            legend_font_size=legend_font_size,
        )
    else:
        print(f"No rollout error curves stored in {data['source_path']}; skipped rollout_errors.png")

    print(f"Paper figures written to {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        raise SystemExit(
            "usage: generate_plots.py [CONFIG.json | plot_data.npz]\n"
            "If omitted, regenerates figures for all default regional-model configs."
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
