"""Learning omega-limit partitions from approximate Koopman invariants.

Library code for the paper "Learning Omega-Limit Partitions from Approximate
Koopman Invariants" (docs/acc_27_multiple-linear-models.tex).

Modules
-------
paths            project root, path resolution, external-dependency import setup
config           JSON config loading shared by the scripts
dictionary       RBF centers and (RBF + polynomial) dictionaries
invariants       generalized eigenvalue problem for approximate invariant observables
partition        graph-merged partition (Algorithm 1): fit, save/load, classify
switched_models  one local EDMD model per partition component; switched rollout
anchors          anchor-based mapping of partition ids to classes for validation
plotting         figures for the partition pipeline
tables           ASCII table printing
legacy           helpers for the earlier single-eigenfunction classifier scripts
"""

from . import paths  # noqa: F401  (sets up imports of external packages)
