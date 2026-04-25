"""Run a notebook-cell-style plot script headlessly, saving to figures/.

The plot scripts in scripts/plot/ were written as Jupyter cells: they call
plt.show() and assume an interactive backend, plus they resolve data files
using bare relative names (e.g. "feature_token_influence_<site>.pt"). This
runner:

  1. chdir's into the data directory (default: reproduction/) so bare-name
     data lookups and glob patterns resolve correctly.
  2. Forces matplotlib Agg backend (no GUI, no display needed).
  3. Monkey-patches plt.show so it saves to figures/<script-stem>[_N].png
     (that is, <data_dir>/figures/<script-stem>.png).
  4. Executes the target plot script in __main__ context.

Usage:
    # from repo root:
    python scripts/plot/run_plot.py scripts/plot/plot_entropy_vs_depth.py

    # override data dir:
    DATA_DIR=. python scripts/plot/run_plot.py scripts/plot/plot_entropy_vs_depth.py

Output:
    <data_dir>/figures/<plot_script_stem>.png   (plus _2, _3, ... if the
    script calls show() more than once)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("usage: python scripts/plot/run_plot.py <plot_script.py>", file=sys.stderr)
    sys.exit(2)

script_path = Path(sys.argv[1]).resolve()
if not script_path.exists():
    print(f"[ERROR] {script_path} not found", file=sys.stderr)
    sys.exit(1)

data_dir = Path(os.environ.get("DATA_DIR", "reproduction")).resolve()
if not data_dir.exists():
    print(
        f"[ERROR] DATA_DIR {data_dir} does not exist. "
        "Override with DATA_DIR=<path> or create reproduction/.",
        file=sys.stderr,
    )
    sys.exit(1)

os.chdir(data_dir)
print(f"[run_plot] cwd -> {data_dir}", flush=True)

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

stem = script_path.stem
_show_count = [0]
_orig_savefig = plt.savefig
_orig_close = plt.close


def _patched_show(*args, **kwargs):
    _show_count[0] += 1
    suffix = "" if _show_count[0] == 1 else f"_{_show_count[0]}"
    out = figures_dir / f"{stem}{suffix}.png"
    _orig_savefig(out, dpi=150, bbox_inches="tight")
    _orig_close()
    print(f"[run_plot] Saved {data_dir / out}", flush=True)


plt.show = _patched_show

print(f"[run_plot] Executing {script_path} headlessly...", flush=True)
src = script_path.read_text()
exec(compile(src, str(script_path), "exec"), {"__name__": "__main__", "__file__": str(script_path)})

if _show_count[0] == 0:
    print(f"[run_plot] WARNING: {stem} did not call plt.show(); no figure saved.", flush=True)
else:
    print(f"[run_plot] Done. {_show_count[0]} figure(s) saved.", flush=True)
