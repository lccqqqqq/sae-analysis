"""Entropy-vs-depth figure for any preset (generalization of the paper's Fig 5).

For a given preset, reads entropy_comparison_*.pt for every layer in
data/<preset>/latest/ (or the given timestamp), aggregates per-feature mean
entropy across batches, and plots:

  - gray scatter : each feature's batch-mean entropy at its layer
  - colored lines: top-N most-activated features tracked across layers
  - blue dashed : mean entropy of the top-M features per layer
  - red dashed  : mean token-vector entropy per layer

Usage:
    python scripts/plotting/plot_entropy_vs_depth_preset.py --preset gemma-2-2b
    python scripts/plotting/plot_entropy_vs_depth_preset.py --preset pythia-70m \\
        --timestamp 20260415_072734
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]


def find_per_layer_files(preset: str, timestamp: str | None) -> dict[int, Path]:
    base = ROOT / "data" / preset
    run_dir = base / timestamp if timestamp else base / "latest"
    if not run_dir.exists():
        raise FileNotFoundError(f"No run dir: {run_dir}")
    files = list(run_dir.glob("entropy_comparison_*layer*.pt"))
    out: dict[int, Path] = {}
    for f in files:
        try:
            layer_idx = int(f.stem.rsplit("layer", 1)[-1])
            out[layer_idx] = f
        except ValueError:
            continue
    if not out:
        raise FileNotFoundError(f"No layer files under {run_dir}")
    return out


def aggregate_layer(file_path: Path) -> dict:
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    feat_ents: dict[int, list[float]] = defaultdict(list)
    feat_acts: dict[int, float] = defaultdict(float)
    tok_ents: list[float] = []
    for br in data["batch_results"]:
        for fid, ent in (br.get("feature_entropies") or {}).items():
            feat_ents[int(fid)].append(float(ent))
        for fid, act in (br.get("feature_activations") or {}).items():
            feat_acts[int(fid)] += float(act)
        tv = br.get("token_vector_entropy")
        if tv is not None:
            tok_ents.append(float(tv))
    return {
        "feat_mean": {fid: float(np.mean(v)) for fid, v in feat_ents.items()},
        "feat_total_activation": dict(feat_acts),
        "token_vector_entropy_mean": float(np.mean(tok_ents)) if tok_ents else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", type=str, required=True)
    ap.add_argument("--timestamp", type=str, default=None)
    ap.add_argument("--n-tracked", type=int, default=10)
    ap.add_argument("--n-top-avg", type=int, default=20)
    style = ap.add_mutually_exclusive_group()
    style.add_argument("--violin", action="store_true",
                       help="Use violin plots per layer instead of gray scatter.")
    style.add_argument("--boxplot", action="store_true",
                       help="Use box plots per layer instead of gray scatter.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = find_per_layer_files(args.preset, args.timestamp)
    layers = sorted(files)
    print(f"[INFO] {args.preset}: {len(layers)} layers ({layers[0]}..{layers[-1]})")

    per_layer = {L: aggregate_layer(files[L]) for L in layers}

    total_act: dict[int, float] = defaultdict(float)
    for L in layers:
        for fid, a in per_layer[L]["feat_total_activation"].items():
            total_act[fid] += a
    tracked = [fid for fid, _ in
               sorted(total_act.items(), key=lambda x: x[1], reverse=True)[:args.n_tracked]]

    width = 9 if len(layers) > 12 else 7
    fig, ax = plt.subplots(figsize=(width, 5.2))

    data_by_layer = [list(per_layer[L]["feat_mean"].values()) for L in layers]
    if args.violin:
        parts = ax.violinplot(data_by_layer, positions=layers,
                              showmeans=False, showmedians=True, widths=0.85)
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue"); pc.set_edgecolor("black"); pc.set_alpha(0.45)
        for key in ("cmins", "cmaxes", "cbars", "cmedians"):
            if key in parts:
                parts[key].set_color("black"); parts[key].set_alpha(0.7)
                parts[key].set_linewidth(0.9)
    elif args.boxplot:
        bp = ax.boxplot(data_by_layer, positions=layers, widths=0.55,
                        showfliers=True, patch_artist=True,
                        flierprops=dict(marker=".", markersize=2,
                                        alpha=0.25, markeredgecolor="gray"),
                        medianprops=dict(color="red", linewidth=1.2),
                        whiskerprops=dict(color="black", linewidth=0.9),
                        capprops=dict(color="black", linewidth=0.9),
                        boxprops=dict(edgecolor="black", linewidth=0.9))
        for box in bp["boxes"]:
            box.set_facecolor("steelblue"); box.set_alpha(0.45)
    else:
        # Gray scatter (default)
        for L in layers:
            vals = list(per_layer[L]["feat_mean"].values())
            ax.scatter([L] * len(vals), vals,
                       color="gray", alpha=0.18, s=8, zorder=1)

    # Blue dashed: mean entropy of the top-M features *by activation* at each layer
    # ("most-pronounced" features, not highest-entropy features).
    top_avg = []
    for L in layers:
        acts = per_layer[L]["feat_total_activation"]      # fid -> sum of last-pos acts
        ents = per_layer[L]["feat_mean"]                  # fid -> mean entropy
        # pick the top-M by activation, then read off their entropies
        top_fids = [fid for fid, _ in sorted(
            acts.items(), key=lambda x: x[1], reverse=True)[:args.n_top_avg]]
        vals = [ents[f] for f in top_fids if f in ents]
        top_avg.append(float(np.mean(vals)) if vals else np.nan)
    ax.plot(layers, top_avg, "b--", linewidth=2,
            label=f"Top-{args.n_top_avg} (by activation) mean", zorder=4)

    # Red dashed: token-vector entropy
    tok = [per_layer[L]["token_vector_entropy_mean"] for L in layers]
    ax.plot(layers, tok, "r--", linewidth=2, label="Token-vector entropy", zorder=4)

    # Colored lines: tracked features
    palette = plt.cm.tab10(np.linspace(0, 1, args.n_tracked))
    for i, fid in enumerate(tracked):
        xs, ys = [], []
        for L in layers:
            if fid in per_layer[L]["feat_mean"]:
                xs.append(L); ys.append(per_layer[L]["feat_mean"][fid])
        if xs:
            ax.plot(xs, ys, "o-", color=palette[i], linewidth=1.3, markersize=4,
                    alpha=0.85, label=f"F{fid}", zorder=3)

    ax.set_xlabel("Layer depth", fontsize=12)
    ax.set_ylabel("Mean feature entropy (bits)", fontsize=12)
    ax.set_title(f"{args.preset} — feature entropy vs layer depth", fontsize=13)
    if len(layers) <= 12:
        ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.85)
    fig.tight_layout()

    style_tag = ("_violin" if args.violin
                 else "_boxplot" if args.boxplot else "")
    out_path = args.out or (ROOT / "plots" /
                            f"entropy_vs_depth_{args.preset}{style_tag}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
