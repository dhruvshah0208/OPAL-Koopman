"""Project paths and path resolution.

The simulator and the EDMD dictionary classes are vendored in this package
(``dynamical_systems`` and ``basis_functions``).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
PAPER_FIGURES_DIR = FIGURES_DIR / "paper"


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths pass through)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path

