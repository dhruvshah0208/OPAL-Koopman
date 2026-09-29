"""Learning omega-limit partitions from approximate Koopman invariants.

Library code for the paper "Learning Omega-Limit Partitions from Approximate
Koopman Invariants".

Modules
-------
paths            project root and path resolution
config           JSON config loading shared by the scripts
dictionary       RBF centers and (RBF + polynomial) dictionaries
invariants       generalized eigenvalue problem for approximate invariant observables
partition        graph-merged partition (Algorithm 1): fit, save/load, classify
regional_models  one regional EDMD model per partition component; regional rollout
anchors          anchor-based mapping of partition ids to classes for validation
plotting         figures for the partition pipeline
tables           ASCII table printing
dynamical_systems  SystemsAutonomous simulator (vendored from src_deep_koopman)
basis_functions    EDMD dictionary classes (vendored from MPV-SPV-pruning-code)
systems          legacy local simulator for bistable_oscillations (superseded by dynamical_systems)
"""

from . import paths  # noqa: F401
