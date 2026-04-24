"""Run a notebook-cell-style plot script headlessly, saving to figures/.

The plot scripts at the repo root were written as Jupyter cells: they call
plt.show() and assume an interactive backend. This runner:
  1. Forces matplotlib Agg backend (no GUI, no display needed).
  2. Monkey-patches plt.show so it saves to figures/<script-stem>[_N].png.
  3. Executes the target script in __main__ context.

Usage:
    python scripts/run_plot.py <plot_script.py>

Output:
    figures/<plot_script_stem>.png  (and _2, _3, ... if it calls show() more
    than once)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("usage: python scripts/run_plot.py <plot_script.py>", file=sys.stderr)
    sys.exit(2)

script_path = Path(sys.argv[1]).resolve()
if not script_path.exists():
    print(f"[ERROR] {script_path} not found", file=sys.stderr)
    sys.exit(1)

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
    print(f"[run_plot] Saved {out}", flush=True)


plt.show = _patched_show

print(f"[run_plot] Executing {script_path} headlessly...", flush=True)
src = script_path.read_text()
exec(compile(src, str(script_path), "exec"), {"__name__": "__main__", "__file__": str(script_path)})

if _show_count[0] == 0:
    print(f"[run_plot] WARNING: {stem} did not call plt.show(); no figure saved.", flush=True)
else:
    print(f"[run_plot] Done. {_show_count[0]} figure(s) saved.", flush=True)
