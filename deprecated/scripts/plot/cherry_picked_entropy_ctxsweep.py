"""Plot H_pos vs CTX_LEN for the cherry-picked feature pilot sweep.

Reads ``ctx<NNN>_<tier>.csv`` files from a single experiment dir and
emits a two-panel figure:

  Left  — per-tier mean H_pos vs CTX_LEN (ALL ok events).
          Shaded band = ±1 std.  log2(ctx_len) reference dashed.
          Reflects the "raw" sweep including the sample-composition effect
          (different events pass the JumpReLU gate at different ctx_lens).

  Right — same axes but restricted to the INTERSECTION of (feature_id,
          top_act_rank) events that are ``ok`` at every swept CTX_LEN.
          Removes the sample-composition confound — same events compared
          across all CTX_LENs — so any remaining trend is intrinsic to
          the gradient pipeline rather than the threshold gate.

Side panel below: per-(tier, ctx_len) ok-count and intersection-count.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

TIER_ORDER = ["token", "phrase", "concept", "abstract"]
TIER_COLORS = {
    "token":    "#1f77b4",
    "phrase":   "#2ca02c",
    "concept":  "#ff7f0e",
    "abstract": "#d62728",
}


def load_sweep(expt_dir: Path) -> tuple[pd.DataFrame, list[int]]:
    rows = []
    ctx_lens: set[int] = set()
    for csv in sorted(expt_dir.glob("ctx*_*.csv")):
        # ctx016_token.csv -> ctx_len=16, tier=token
        stem = csv.stem
        ctx_part, tier_part = stem.split("_", 1)
        ctx_len = int(ctx_part[3:])
        tier = tier_part
        if tier not in TIER_ORDER:
            continue
        df = pd.read_csv(csv)
        df["_ctx_len"] = ctx_len
        rows.append(df)
        ctx_lens.add(ctx_len)
    if not rows:
        raise FileNotFoundError(f"no ctx*_*.csv under {expt_dir}")
    df = pd.concat(rows, ignore_index=True)
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)
    return df, sorted(ctx_lens)


def _agg_mean_std(df: pd.DataFrame, tier: str, ctx_lens: list[int]):
    means, stds, ns = [], [], []
    for n in ctx_lens:
        sub = df[(df["tier"] == tier) & (df["_ctx_len"] == n)
                 & (df["status"] == "ok")]
        means.append(sub["H_pos_bits"].mean() if len(sub) else float("nan"))
        stds.append(sub["H_pos_bits"].std() if len(sub) > 1 else 0.0)
        ns.append(len(sub))
    return np.array(means), np.array(stds), np.array(ns)


def plot_one_panel(ax, df: pd.DataFrame, ctx_lens: list[int], title: str,
                   show_log2_ref: bool = True):
    for tier in TIER_ORDER:
        means, stds, ns = _agg_mean_std(df, tier, ctx_lens)
        c = TIER_COLORS[tier]
        ax.plot(ctx_lens, means, "o-", color=c, label=tier, linewidth=2,
                markersize=7, zorder=3)
        ax.fill_between(ctx_lens, means - stds, means + stds, color=c,
                        alpha=0.15, zorder=1)
    if show_log2_ref:
        x = np.array(ctx_lens)
        ax.plot(x, np.log2(x), "k--", alpha=0.55,
                label="log2(ctx_len)", zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ctx_lens)
    ax.set_xticklabels(ctx_lens)
    ax.set_xlabel("ctx_len (tokens, log2)", fontsize=11)
    ax.set_ylabel("H_pos (bits)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")


def restrict_to_intersection(df: pd.DataFrame, ctx_lens: list[int]) -> pd.DataFrame:
    """Keep only (feature_id, top_act_rank) pairs that are ok at every ctx_len."""
    ok = df[df["status"] == "ok"].copy()
    pivot = (
        ok.assign(_one=1)
          .pivot_table(index=["feature_id", "top_act_rank"],
                       columns="_ctx_len", values="_one",
                       aggfunc="sum", fill_value=0)
    )
    keep = pivot[(pivot[ctx_lens] > 0).all(axis=1)].index
    keep_set = set(map(tuple, keep.tolist()))
    mask = df.apply(lambda r: (int(r["feature_id"]), int(r["top_act_rank"]))
                    in keep_set, axis=1)
    return df[mask & (df["status"] == "ok")].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expt-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    expt_dir = args.expt_dir.resolve()

    df, ctx_lens = load_sweep(expt_dir)
    n_total = len(df)
    n_ok = (df["status"] == "ok").sum()
    print(f"[INFO] total rows: {n_total}, ok: {n_ok}, ctx_lens: {ctx_lens}")

    df_intersect = restrict_to_intersection(df, ctx_lens)
    n_intersect_pairs = (
        df_intersect.groupby(["feature_id", "top_act_rank"]).ngroups
    )
    n_intersect_rows = len(df_intersect)
    print(f"[INFO] intersection events: {n_intersect_pairs} pairs, "
          f"{n_intersect_rows} rows")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_one_panel(axes[0], df, ctx_lens,
                   f"All ok events  (n varies per ctx_len)")
    plot_one_panel(axes[1], df_intersect, ctx_lens,
                   f"Intersection  ({n_intersect_pairs} events ok at every ctx_len)")

    manifest_path = expt_dir / "manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        sub = (f"{m.get('preset')} layer {m.get('layer')}  "
               f"({m.get('sae_id')}, top_k={m.get('top_k')}, "
               f"variant={m.get('variant', 'unknown')})")
    else:
        sub = ""
    fig.suptitle(f"H_pos vs CTX_LEN  —  {sub}", fontsize=12,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    output = args.output or (expt_dir / "figures" / "h_pos_vs_ctxlen.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {output}")

    # Also print the wide-table summary for both views.
    print("\n=== ALL ok events: mean ± std (n) ===")
    print(f"{'tier':>10}  " + "  ".join(f"ctx{n:03d}" for n in ctx_lens))
    for tier in TIER_ORDER:
        means, stds, ns = _agg_mean_std(df, tier, ctx_lens)
        cells = [f"{m:.2f}±{s:.2f}({n:d})" for m, s, n in zip(means, stds, ns)]
        print(f"{tier:>10}  " + "  ".join(cells))

    print("\n=== INTERSECTION ok events: mean ± std (n) ===")
    print(f"{'tier':>10}  " + "  ".join(f"ctx{n:03d}" for n in ctx_lens))
    for tier in TIER_ORDER:
        means, stds, ns = _agg_mean_std(df_intersect, tier, ctx_lens)
        cells = [f"{m:.2f}±{s:.2f}({n:d})" for m, s, n in zip(means, stds, ns)]
        print(f"{tier:>10}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
