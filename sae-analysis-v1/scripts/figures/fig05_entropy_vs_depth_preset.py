"""
Fig 5 (generalized): Feature entropy vs layer depth, one violin per preset.

Reads the new data layout:
    data/<preset>/<timestamp>/entropy_comparison_<site>.pt    (one file per layer)
and walks all layers present. Picks the most recent run directory by default
(via the 'latest' symlink) but can be pointed at a specific timestamp.

Output:
    plots/entropy_vs_depth__<preset>.png    (one file per preset)

Usage:
    python scripts/figures/fig05_entropy_vs_depth_preset.py --preset gpt2-small
    python scripts/figures/fig05_entropy_vs_depth_preset.py --preset pythia-70m llama-3.2-1b
    python scripts/figures/fig05_entropy_vs_depth_preset.py --all   # all presets with data
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "plots"
N_TOP_AVG = 20


def find_run_dir(preset, run_timestamp=None):
    preset_dir = ROOT / "data" / preset
    if run_timestamp:
        return preset_dir / run_timestamp
    # Prefer the run with the most layer files (avoids accidentally picking
    # a stale smoke symlink when the full run has already finished in a
    # sibling directory).
    candidates = [p for p in preset_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories under {preset_dir}")
    def n_layers(d):
        return len(list(d.glob("entropy_comparison_*.pt")))
    best = max(candidates, key=lambda d: (n_layers(d), d.stat().st_mtime))
    return best


def load_preset(preset, run_timestamp=None):
    run_dir = find_run_dir(preset, run_timestamp)
    files = sorted(run_dir.glob("entropy_comparison_*.pt"))
    if not files:
        raise FileNotFoundError(f"No entropy_comparison files in {run_dir}")
    layers = []
    feat_by_layer = {}    # layer -> list of per-feature mean entropies
    freq_by_layer = {}    # layer -> dict{feat_idx -> (mean_entropy, n_batches_active)}
    token_by_layer = {}   # layer -> mean token-vector entropy
    for f in files:
        m = re.search(r"layer(\d+)\.pt$", f.name)
        if not m:
            continue
        layer = int(m.group(1))
        d = torch.load(f, map_location="cpu", weights_only=False)
        # Per-feature entropy list across batches + activation count.
        per_feat = {}
        token_ents = []
        for br in d["batch_results"]:
            for feat_idx, ent in br.get("feature_entropies", {}).items():
                per_feat.setdefault(feat_idx, []).append(ent)
            if br.get("token_vector_entropy") is not None:
                token_ents.append(br["token_vector_entropy"])
        feat_by_layer[layer] = [float(np.mean(v)) for v in per_feat.values()]
        freq_by_layer[layer] = {
            fi: (float(np.mean(v)), len(v)) for fi, v in per_feat.items()
        }
        token_by_layer[layer] = float(np.mean(token_ents)) if token_ents else np.nan
        layers.append(layer)
    layers.sort()
    meta = {
        "preset": preset,
        "run_dir": str(run_dir),
        "num_batches": d["config"].get("batch_size", "?") and len(d["batch_results"]),
        "sae_arch": d["config"].get("sae_arch", "?"),
    }
    return layers, feat_by_layer, freq_by_layer, token_by_layer, meta


def plot_one(preset, run_timestamp=None):
    layers, feat_by_layer, freq_by_layer, token_by_layer, meta = load_preset(
        preset, run_timestamp
    )

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(layers) + 4), 5))

    data = [feat_by_layer[L] for L in layers]
    parts = ax.violinplot(data, positions=layers, widths=0.7,
                          showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("lightgray")
        pc.set_edgecolor("gray")
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    # Blue dashed: mean entropy of the top-N features ranked by
    # activation frequency (number of batches each feature fired in).
    # Ties are broken by higher mean activation-entropy to keep the choice deterministic.
    top_avgs = []
    for L in layers:
        ranked = sorted(
            freq_by_layer[L].items(),
            key=lambda kv: (kv[1][1], kv[1][0]),   # (n_active, mean_entropy)
            reverse=True,
        )
        top_ents = [ent for _, (ent, _n) in ranked[:N_TOP_AVG]]
        top_avgs.append(np.mean(top_ents) if top_ents else np.nan)
    ax.plot(layers, top_avgs, "b--", linewidth=2,
            label=f"Avg top-{N_TOP_AVG} most-frequent features", zorder=4)

    # Red dashed: token-vector entropy per layer.
    tok = [token_by_layer[L] for L in layers]
    ax.plot(layers, tok, "r--", linewidth=2, label="Token vector entropy", zorder=4)

    ax.set_xlabel("Layer depth", fontsize=12)
    ax.set_ylabel("Entropy (bits)", fontsize=12)
    ax.set_title(f"Feature entropy vs depth — {preset} "
                 f"({meta['sae_arch']} SAE, {meta['num_batches']} batches)",
                 fontsize=12)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{L}" for L in layers], fontsize=9)
    ax.axhline(np.log2(64), color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(layers[-1], np.log2(64) - 0.1, f"max = log₂(64) = {np.log2(64):.2f}",
            fontsize=8, color="gray", ha="right", va="top")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"entropy_vs_depth__{preset}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {preset}: {len(layers)} layers  ->  {out.relative_to(ROOT)}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", nargs="+", default=None,
                        help="Preset names (default: use --all).")
    parser.add_argument("--all", action="store_true",
                        help="Plot every preset that has data in data/<preset>/.")
    parser.add_argument("--run-timestamp", default=None,
                        help="Specific run timestamp instead of most-layers auto-pick.")
    args = parser.parse_args()

    if args.all or not args.preset:
        data_root = ROOT / "data"
        presets = sorted(p.name for p in data_root.iterdir()
                         if p.is_dir() and list(p.glob("*/entropy_comparison_*.pt")))
    else:
        presets = args.preset

    if not presets:
        print("No presets with entropy_comparison data found.")
        return

    for preset in presets:
        try:
            plot_one(preset, args.run_timestamp)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ✗ {preset}: {e}")


if __name__ == "__main__":
    main()
