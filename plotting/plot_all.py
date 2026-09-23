"""Regenerate every paper figure from the cached arrays for one or both systems.

    python plotting/plot_all.py                 # duffing and bistable_oscillations
    python plotting/plot_all.py duffing --show
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ("plot_partition.py", "plot_validation.py", "plot_rollouts.py", "plot_rollout_errors.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("systems", nargs="*", default=["duffing", "bistable_oscillations"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: the plotting folder)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    for system in args.systems:
        extra = []
        if args.data_dir is not None:
            extra += ["--data-dir", str(args.data_dir)]
        if args.out is not None:
            extra += ["--out", str(args.out)]
        if args.show:
            extra.append("--show")
        for script in SCRIPTS:
            subprocess.run([sys.executable, str(HERE / script), system, *extra], check=True)
