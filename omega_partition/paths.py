"""Project paths and import setup for the external dependencies.

The library depends on two packages that live outside this repository and are
installed (editable) in the ``macbook_env`` conda environment:

* ``src``               -- basis functions and pruning (MPV-SPV-pruning-code)
* ``src_deep_koopman``  -- ``SystemsAutonomous`` dynamical-system simulator

``src_deep_koopman.dynamical_systems`` does ``from utils import ts``, which only
resolves when the package directory itself is on ``sys.path``; ``_setup_external_imports``
takes care of that so every script does not have to.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths pass through)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _setup_external_imports() -> None:
    spec = importlib.util.find_spec("src_deep_koopman")
    if spec is None or not spec.submodule_search_locations:
        return
    for location in spec.submodule_search_locations:
        if location not in sys.path:
            sys.path.append(location)


_setup_external_imports()
