# Learning Omega-Limit Partitions from Approximate Koopman Invariants

Code for the paper in [`docs/acc_27_multiple-linear-models.tex`](docs/acc_27_multiple-linear-models.tex).
Given snapshot pairs of a nonlinear system with several attractors, the pipeline

1. solves a generalized eigenvalue problem for *approximate Koopman invariant observables*,
2. clusters the data in those coordinates and merges the fine cells into an approximate
   **omega-limit partition** (Algorithm 1), and
3. fits one local EDMD model per partition cell and evaluates switched long-horizon prediction.

## Layout

```
omega_partition/     library (invariants, partition, switched models, anchors, plotting)
scripts/             one script per pipeline stage; each takes one JSON config
scripts/legacy/      earlier single-eigenfunction classifier + pruning (not used in the paper)
configs/<system>/    JSON configs for duffing, bistable_oscillations, bistable (1-D toy)
data/<system>/       datasets and pipeline outputs (*.npz are git-ignored; *.meta.json tracked)
figures/             figures written by the scripts
docs/                paper source, bibliography, paper figures
notebooks/           exploratory notebooks (use the external `src` package directly)
```

## Setup

```bash
conda activate macbook_env
```

The library needs two packages that are editable-installed in that environment and live
outside this repository: `src` (basis functions, from `../MPV-SPV-pruning-code`) and
`src_deep_koopman` (`SystemsAutonomous` simulator, from
`~/Documents/PhD-code-files/Benchmarking_Deep_Koopman`). Everything else is numpy / scipy /
scikit-learn / matplotlib. Run all commands from the repository root; set
`MPLBACKEND=Agg` when there is no display.

## Running the pipeline

Replace `<system>` with `duffing` or `bistable_oscillations` (paper examples) or `bistable`.
The datasets under `data/` are already generated, so you can start at step 3.

```bash
# 1. training snapshots + K-means RBF centers        -> data/<system>/<system>_data.npz
python scripts/generate_data.py            configs/duffing/generate_duffing_data.json
# 2. labeled validation trajectories                 -> data/<system>/<system>_validation_data.npz
python scripts/generate_validation_data.py configs/<system>/generate_validation_data.json
# 3. approximate omega-limit partition (Algorithm 1) -> data/<system>/partition_output.npz
python scripts/build_partition.py          configs/<system>/build_partition.json
# 4. partition accuracy against anchor states
python scripts/validate_partition.py       configs/<system>/validate_partition.json
# 5. local EDMD model per cell, rollout metrics, figures -> data/<system>/switched_models.npz, figures/<system>/
python scripts/build_multiple_linear_models.py configs/<system>/build_multiple_linear_models.json
# 6. paper-styled figures from the saved plot data  -> docs/figures/<system>/
python scripts/generate_plots.py
python scripts/generate_merge_diagram.py           # Fig. 1 schematic
```

For `bistable_oscillations` the data-generation configs are `generate_data.json`,
`generate_target_data{1,2}.json` (reference limit cycles used as anchors) and
`generate_validation_data.json`. Step 5 prints the numbers in Table I of the paper
(classification accuracy and mean relative rollout error at selected time steps).

Every config has an `"explanation"` block documenting its keys; paths inside configs are
relative to the repository root.

## Notes

- The first approximate-invariant mode is the constant function whenever the dictionary
  contains the constant (`d_poly >= 0`). Its `scalar_cluster_count` entry must be `1`
  (splitting a constant clusters floating-point noise and is not reproducible).
- Gaussian RBFs are `exp(-gamma * ||x - c||^2)`: larger `gamma` means narrower bumps.
- `bistable_oscillations` is not registered in the installed `src_deep_koopman`, so its
  datasets cannot currently be regenerated; the saved `.npz` files are sufficient for
  steps 3–6 (phase plots then fall back to sampled validation trajectories).
