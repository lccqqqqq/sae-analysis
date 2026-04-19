"""
Figure 5 — Feature entropy versus layer depth (layers 0–5).
Violin of per-feature mean entropy at each layer, with aggregate summary lines.
Per-feature trajectories across layers are NOT drawn: SAEs at different layers
are trained independently, so feature index f at layer 0 and layer 5 are
unrelated directions in different dictionaries.

Prerequisite data:
    entropy_comparison_resid_out_layer{0..5}_<timestamp>.pt  (one file per layer)
    (produced by: python scripts/analysis/compare_entropies_multi_layer.py --layers 0 1 2 3 4 5)

Output:
    paper/figures/entropy_vs_depth.png
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = ROOT / "plots/entropy_vs_depth.png"

LAYERS = [0, 1, 2, 3, 4, 5]
N_TOP_AVG = 20   # Features used for the blue average line


def find_latest(layer):
    pattern = str(ROOT / "data" / f"entropy_comparison_resid_out_layer{layer}_*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No entropy_comparison file for layer {layer}. "
                                f"Run: python scripts/analysis/compare_entropies_multi_layer.py "
                                f"--layers {' '.join(map(str, LAYERS))}")
    return max(files, key=lambda f: Path(f).stat().st_mtime)


def main():
    # Load all layer data
    layer_data = {}
    for layer in LAYERS:
        path = find_latest(layer)
        layer_data[layer] = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  Layer {layer}: {Path(path).name}")

    # Collect per-feature mean entropy and token vector entropy per layer
    feat_entropies = defaultdict(dict)  # layer -> feat_idx -> mean entropy
    token_entropies = {}  # layer -> mean token vector entropy

    for layer in LAYERS:
        batch_results = layer_data[layer]["batch_results"]
        token_ents = []
        for br in batch_results:
            for feat_idx, ent in br.get("feature_entropies", {}).items():
                if feat_idx not in feat_entropies[layer]:
                    feat_entropies[layer][feat_idx] = []
                feat_entropies[layer][feat_idx].append(ent)
            if br.get("token_vector_entropy") is not None:
                token_ents.append(br["token_vector_entropy"])
        feat_entropies[layer] = {k: float(np.mean(v)) for k, v in feat_entropies[layer].items()}
        token_entropies[layer] = float(np.mean(token_ents)) if token_ents else np.nan

    fig, ax = plt.subplots(figsize=(8, 5))

    # Violin plot: per-feature mean entropy distribution at each layer
    data_per_layer = [list(feat_entropies[layer].values()) for layer in LAYERS]
    parts = ax.violinplot(data_per_layer, positions=LAYERS, widths=0.7,
                          showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("lightgray")
        pc.set_edgecolor("gray")
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    # Blue dashed: average of top-N features per layer
    avg_ents = []
    for layer in LAYERS:
        vals = sorted(feat_entropies[layer].values(), reverse=True)[:N_TOP_AVG]
        avg_ents.append(np.mean(vals) if vals else np.nan)
    ax.plot(LAYERS, avg_ents, "b--", linewidth=2, label=f"Avg top-{N_TOP_AVG} features", zorder=4)

    # Red dashed: token vector entropy
    tok_vals = [token_entropies.get(l, np.nan) for l in LAYERS]
    ax.plot(LAYERS, tok_vals, "r--", linewidth=2, label="Token vector entropy", zorder=4)

    ax.set_xlabel("Layer depth", fontsize=12)
    ax.set_ylabel("Entropy (bits)", fontsize=12)
    ax.set_title("Feature entropy vs layer depth (violin)", fontsize=13)
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS], fontsize=9)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
