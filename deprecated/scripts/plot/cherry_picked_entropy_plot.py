"""Visualise H_pos for the cherry-picked feature pilot.

Reads the per-tier CSVs produced by
``scripts/analysis/topact_entropy.py`` (one row per (feature, top-act)
event) and emits a two-panel figure:

  Left panel  — per-tier distribution of H_pos as a violin + strip plot
                with each event as a jittered point. The horizontal
                dashed line is log2(ctx_len), the maximal entropy of the
                positional-influence distribution.
  Right panel — per-feature swarm. One column per feature (ordered by
                tier then ascending mean H_pos), 16 jittered points
                showing the event distribution, a horizontal segment at
                the per-feature mean, and the feature_id labelled below.
                Tier shown via x-axis colour bar.

Usage:
    python scripts/plot/cherry_picked_entropy_plot.py \\
        --expt-dir data/cherry_picked_feature_entropy/latest
"""

from __future__ import annotations

import argparse
import json
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


def load_events(expt_dir: Path, ctx_len: int) -> pd.DataFrame:
    frames = []
    for tier in TIER_ORDER:
        path = expt_dir / f"ctx{ctx_len:03d}_{tier}.csv"
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            f"no ctx{ctx_len:03d}_*.csv under {expt_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)
    return df


def plot_tier_distribution(ax, ok: pd.DataFrame, ctx_len: int):
    rng = np.random.default_rng(0)
    parts = ax.violinplot(
        [ok.loc[ok.tier == t, "H_pos_bits"].values for t in TIER_ORDER],
        positions=range(len(TIER_ORDER)),
        showmeans=False, showmedians=True, widths=0.7,
    )
    for body, tier in zip(parts["bodies"], TIER_ORDER):
        body.set_facecolor(TIER_COLORS[tier])
        body.set_edgecolor("black")
        body.set_alpha(0.35)
    for line in ("cmedians", "cmins", "cmaxes", "cbars"):
        if line in parts:
            parts[line].set_color("black")
            parts[line].set_linewidth(1.0)

    for i, tier in enumerate(TIER_ORDER):
        vals = ok.loc[ok.tier == tier, "H_pos_bits"].values
        if len(vals) == 0:
            continue
        x = i + rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x, vals, s=14, color=TIER_COLORS[tier],
                   edgecolor="black", linewidth=0.3, alpha=0.75, zorder=3)
        mean = float(vals.mean())
        ax.hlines(mean, i - 0.32, i + 0.32, colors="black",
                  linewidth=2.0, zorder=4)
        ax.text(i + 0.34, mean, f"{mean:.2f}", va="center", fontsize=9)

    ax.axhline(np.log2(ctx_len), color="black", linestyle="--",
               alpha=0.55, label=f"log2(ctx_len)={np.log2(ctx_len):.2f}")
    ax.set_xticks(range(len(TIER_ORDER)))
    ax.set_xticklabels(TIER_ORDER)
    ax.set_xlabel("tier", fontsize=11)
    ax.set_ylabel("H_pos (bits)", fontsize=11)
    ax.set_title("Per-tier distribution of H_pos", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")


def plot_per_feature(ax, ok: pd.DataFrame, ctx_len: int, manifest: dict):
    # Order: tier ascending (token, phrase, concept, abstract), then mean asc.
    per_feat = (
        ok.groupby(["tier", "feature_id"], observed=True)
          .agg(mean=("H_pos_bits", "mean"),
               std=("H_pos_bits", "std"),
               count=("H_pos_bits", "size"))
          .reset_index()
    )
    # Stable sort by (tier order index, mean).
    per_feat["tier_rank"] = per_feat["tier"].map({t: i for i, t in enumerate(TIER_ORDER)})
    per_feat = per_feat.sort_values(["tier_rank", "mean"]).reset_index(drop=True)

    desc_by_id = {int(e["feature_id"]): e.get("description", "")
                  for e in manifest["features"]}

    n_feat = len(per_feat)
    # Adapt point and tick density for large feature counts.
    if n_feat <= 30:
        scatter_size, edge_lw, mean_lw = 18, 0.3, 1.6
        jitter, mean_halfwidth = 0.22, 0.34
    elif n_feat <= 80:
        scatter_size, edge_lw, mean_lw = 8, 0.0, 1.0
        jitter, mean_halfwidth = 0.30, 0.40
    else:
        scatter_size, edge_lw, mean_lw = 4, 0.0, 0.6
        jitter, mean_halfwidth = 0.32, 0.44

    rng = np.random.default_rng(1)
    xticklabels = []
    all_y_values: list[float] = []
    for x, row in per_feat.iterrows():
        fid = int(row["feature_id"])
        tier = row["tier"]
        sub = ok[ok["feature_id"] == fid]["H_pos_bits"].values
        all_y_values.extend(sub.tolist())
        jx = x + rng.uniform(-jitter, jitter, size=len(sub))
        ax.scatter(jx, sub, s=scatter_size, color=TIER_COLORS[tier],
                   edgecolor="black", linewidth=edge_lw,
                   alpha=0.7, zorder=3)
        ax.hlines(row["mean"], x - mean_halfwidth, x + mean_halfwidth,
                  colors="black", linewidth=mean_lw, zorder=4)
        xticklabels.append(f"{fid}\n({tier[0]})")

    # Adaptive y-range: cover the data, the log2(ctx_len) reference line,
    # and a small margin for the tier labels at the top.
    data_min = min(all_y_values) if all_y_values else 0.0
    data_max = max(all_y_values) if all_y_values else 1.0
    log2_ref = float(np.log2(ctx_len))
    y_top = max(data_max, log2_ref) + 0.6   # margin for tier labels
    y_bottom = max(0.0, data_min - 0.3)
    ax.set_ylim(bottom=y_bottom, top=y_top)
    label_y = y_top - 0.15

    # Tier-band shading on the x-axis.
    boundaries = []
    cur = None
    for i, tier in enumerate(per_feat["tier"]):
        if tier != cur:
            boundaries.append(i)
            cur = tier
    boundaries.append(len(per_feat))
    for k in range(len(boundaries) - 1):
        x0 = boundaries[k] - 0.5
        x1 = boundaries[k + 1] - 0.5
        tier = per_feat["tier"].iloc[boundaries[k]]
        ax.axvspan(x0, x1, color=TIER_COLORS[tier], alpha=0.07, zorder=0)
        ax.text((x0 + x1) / 2, label_y, tier,
                ha="center", va="top", fontsize=10,
                color=TIER_COLORS[tier], fontweight="bold")

    ax.axhline(log2_ref, color="black", linestyle="--",
               alpha=0.55, label=f"log2(ctx_len)={log2_ref:.2f}")

    # Adaptive x-tick density: at most ~50 visible labels.
    if n_feat <= 50:
        tick_step = 1
        tick_fontsize = 8
    else:
        tick_step = max(1, n_feat // 50)
        tick_fontsize = 6
    tick_positions = list(range(0, n_feat, tick_step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([xticklabels[i] for i in tick_positions],
                       fontsize=tick_fontsize)
    ax.set_xlabel("feature_id  (tier letter in parentheses)", fontsize=11)
    ax.set_ylabel("H_pos (bits)", fontsize=11)
    ax.set_title(f"Per-feature H_pos  ({n_feat} features, black bar = mean)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # Print per-feature summary alongside the figure (handy for the user).
    print("\n[per-feature summary, sorted by tier then mean H_pos]")
    print(f"{'feature_id':>10}  {'tier':>8}  {'n':>3}  {'mean':>5}  {'std':>5}  description")
    for _, row in per_feat.iterrows():
        fid = int(row["feature_id"])
        d = desc_by_id.get(fid, "")
        print(f"{fid:>10}  {row['tier']:>8}  {int(row['count']):>3}  "
              f"{row['mean']:>5.2f}  {row['std']:>5.2f}  {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expt-dir", type=Path, required=True,
                    help="Experiment dir under "
                         "data/cherry_picked_feature_entropy/")
    ap.add_argument("--ctx-len", type=int, default=None,
                    help="Which ctx_len to plot (e.g. 64). Defaults to "
                         "manifest['ctx_len'] if present.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output PNG path; defaults to "
                         "<expt-dir>/figures/h_pos_overview_ctx<NNN>.png")
    args = ap.parse_args()

    expt_dir = args.expt_dir.resolve()
    manifest = json.loads((expt_dir / "manifest.json").read_text())
    ctx_len = int(args.ctx_len if args.ctx_len is not None
                  else manifest.get("ctx_len", 64))

    df = load_events(expt_dir, ctx_len)
    ok = df[df["status"] == "ok"].copy()
    n_total, n_ok = len(df), len(ok)
    print(f"[INFO] events: {n_total} total, {n_ok} ok, "
          f"{n_total - n_ok} skipped", flush=True)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6.5),
                                     gridspec_kw={"width_ratios": [1.0, 2.4]})
    plot_tier_distribution(ax_l, ok, ctx_len)
    plot_per_feature(ax_r, ok, ctx_len, manifest)

    fig.suptitle(
        f"H_pos on top-{int(manifest['top_k'])} activating contexts  —  "
        f"{manifest['preset']} layer {manifest['layer']}  "
        f"({manifest['sae_id']}, ctx_len={ctx_len}, "
        f"{n_ok}/{n_total} events ok)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    output = args.output or (expt_dir / "figures"
                             / f"h_pos_overview_ctx{ctx_len:03d}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {output}")


if __name__ == "__main__":
    main()
