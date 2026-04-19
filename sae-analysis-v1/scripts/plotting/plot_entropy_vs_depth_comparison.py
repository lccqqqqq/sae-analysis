"""Cross-model entropy-vs-depth comparison — one panel per model.

Small-multiples grid; each panel is the same plot style as
plot_entropy_vs_depth_preset.py (gray scatter + blue top-M mean + red
token-vector) for one model.

Usage:
    python scripts/plotting/plot_entropy_vs_depth_comparison.py \\
        --presets pythia-70m gpt2-small qwen2-0.5b llama-3.2-1b gemma-2-2b
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]


def find_per_layer_files(preset: str, timestamp: str | None = None) -> dict[int, Path]:
    base = ROOT / "data" / preset
    run_dir = base / timestamp if timestamp else base / "latest"
    if not run_dir.exists():
        raise FileNotFoundError(f"No run dir: {run_dir}")
    files = list(run_dir.glob("entropy_comparison_*layer*.pt"))
    out: dict[int, Path] = {}
    for f in files:
        try:
            out[int(f.stem.rsplit("layer", 1)[-1])] = f
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
    ap.add_argument("--presets", type=str, nargs="+", required=True)
    ap.add_argument("--n-top-avg", type=int, default=20)
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--share-y", action="store_true",
                    help="Share y-axis range across panels.")
    style = ap.add_mutually_exclusive_group()
    style.add_argument("--violin", action="store_true",
                       help="Use violin plots per layer instead of gray scatter.")
    style.add_argument("--boxplot", action="store_true",
                       help="Use box plots per layer instead of gray scatter.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        style_tag = ("_violin" if args.violin
                     else "_boxplot" if args.boxplot else "")
        args.out = ROOT / "plots" / f"entropy_vs_depth_crossmodel_grid{style_tag}.png"

    data: dict[str, dict] = {}
    global_ymax = 0.0
    for preset in args.presets:
        files = find_per_layer_files(preset)
        layers = sorted(files)
        per_layer = {L: aggregate_layer(files[L]) for L in layers}
        all_vals = [v for L in layers for v in per_layer[L]["feat_mean"].values()]
        if all_vals:
            global_ymax = max(global_ymax, float(np.max(all_vals)))
        data[preset] = {"layers": layers, "per_layer": per_layer,
                        "n_layers": len(layers)}
        print(f"[INFO] {preset}: {len(layers)} layers")

    n = len(args.presets)
    ncols = min(args.ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.4 * nrows),
                             sharex=False, sharey=args.share_y,
                             squeeze=False)

    global_ymax = math.ceil(global_ymax * 1.05 * 10) / 10

    for idx, preset in enumerate(args.presets):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        pl = data[preset]
        layers = pl["layers"]
        per_layer = pl["per_layer"]

        data_by_layer = [list(per_layer[L]["feat_mean"].values()) for L in layers]
        if args.violin:
            width = min(0.85, 10.0 / max(1, len(layers)))
            parts = ax.violinplot(data_by_layer, positions=layers,
                                  showmeans=False, showmedians=True, widths=width)
            for pc in parts["bodies"]:
                pc.set_facecolor("steelblue"); pc.set_edgecolor("black")
                pc.set_alpha(0.45)
            for key in ("cmins", "cmaxes", "cbars", "cmedians"):
                if key in parts:
                    parts[key].set_color("black"); parts[key].set_alpha(0.7)
                    parts[key].set_linewidth(0.8)
        elif args.boxplot:
            width = min(0.6, 7.0 / max(1, len(layers)))
            bp = ax.boxplot(data_by_layer, positions=layers, widths=width,
                            showfliers=True, patch_artist=True,
                            flierprops=dict(marker=".", markersize=1.5,
                                            alpha=0.2, markeredgecolor="gray"),
                            medianprops=dict(color="red", linewidth=1.0),
                            whiskerprops=dict(color="black", linewidth=0.7),
                            capprops=dict(color="black", linewidth=0.7),
                            boxprops=dict(edgecolor="black", linewidth=0.7))
            for box in bp["boxes"]:
                box.set_facecolor("steelblue"); box.set_alpha(0.45)
        else:
            # Gray scatter
            for L in layers:
                vals = list(per_layer[L]["feat_mean"].values())
                ax.scatter([L] * len(vals), vals,
                           color="gray", alpha=0.18, s=6, zorder=1)

        # Blue dashed: mean entropy of top-M features *by activation* (not entropy).
        top_avg = []
        for L in layers:
            acts = per_layer[L]["feat_total_activation"]
            ents = per_layer[L]["feat_mean"]
            top_fids = [fid for fid, _ in sorted(
                acts.items(), key=lambda x: x[1], reverse=True)[:args.n_top_avg]]
            vals = [ents[f] for f in top_fids if f in ents]
            top_avg.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(layers, top_avg, "b--", linewidth=1.8,
                label=f"Top-{args.n_top_avg} (by act)", zorder=4)

        # Red dashed: token-vector entropy
        tok = [per_layer[L]["token_vector_entropy_mean"] for L in layers]
        ax.plot(layers, tok, "r--", linewidth=1.8,
                label="Token vec.", zorder=4)

        # log2(64) = 6 reference
        ax.axhline(6.0, color="gray", linestyle=":", linewidth=0.8,
                   alpha=0.5, zorder=0)

        ax.set_xlabel("Layer depth", fontsize=10)
        ax.set_ylabel("Mean entropy (bits)", fontsize=10)
        ax.set_title(f"{preset}  (n_L={pl['n_layers']})",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if args.share_y:
            ax.set_ylim(0, global_ymax)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.85)

        # x ticks: only every Nth layer if many layers.
        # (Must set AFTER boxplot/violin since those reset tick labels.)
        if pl["n_layers"] > 14:
            step = max(1, pl["n_layers"] // 7)
            ticks = layers[::step]
        else:
            ticks = layers
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r][c].axis("off")

    fig.suptitle("Feature influence entropy vs layer depth — cross-model comparison",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
