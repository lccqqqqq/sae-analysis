"""Scatter plots: per-feature mean influence entropy H_feat vs feature-intrinsic
"level" proxies (H_vocab, H_logit), one panel per layer.

H_feat for feature a = mean over batches where a was active (above the
compare_entropies.py threshold of 0.2) of its per-batch influence entropy.
H_vocab and H_logit are loaded from data/<preset>/feature_level_entropies.pt.

Outputs:
    plots/feature_level_vs_influence_entropy_{preset}.png   -- 6x2 grid

Usage:
    python scripts/plotting/plot_feature_level_vs_entropy.py \
        --preset pythia-70m --timestamp 20260414_053350
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]


def find_entropy_comparison_file(preset: str, layer: int, timestamp: str | None) -> Path:
    """Locate the entropy_comparison_*.pt for (preset, layer, timestamp)."""
    candidates = []
    if timestamp:
        # Old layout (data/entropy_comparison_resid_out_layer{N}_<ts>.pt)
        p = ROOT / "data" / f"entropy_comparison_resid_out_layer{layer}_{timestamp}.pt"
        if p.exists():
            candidates.append(p)
        # New layout (data/<preset>/<ts>/entropy_comparison_<site>.pt)
        p2 = ROOT / "data" / preset / timestamp / f"entropy_comparison_resid_out_layer{layer}.pt"
        if p2.exists():
            candidates.append(p2)
    if not candidates:
        # Fall back to most recent across either layout
        old = list((ROOT / "data").glob(
            f"entropy_comparison_resid_out_layer{layer}_*.pt"))
        new = list((ROOT / "data" / preset).glob(
            f"*/entropy_comparison_resid_out_layer{layer}.pt"))
        candidates = old + new
    if not candidates:
        raise FileNotFoundError(
            f"No entropy_comparison file for preset={preset}, layer={layer}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def aggregate_feature_entropy(file_path: Path) -> dict[int, float]:
    """Load one entropy_comparison file; return mean H_feat per feature."""
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    bucket: dict[int, list[float]] = {}
    for br in data["batch_results"]:
        for fid, ent in (br.get("feature_entropies") or {}).items():
            bucket.setdefault(int(fid), []).append(float(ent))
    return {fid: float(np.mean(v)) for fid, v in bucket.items()}


def make_scatter(ax, x: np.ndarray, y: np.ndarray, title: str,
                 xlabel: str, ylabel: str = "H_feat (bits)"):
    """One scatter panel with Pearson + Spearman in the title."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 5:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        ax.set_title(title); return
    rho_p, p_p = pearsonr(x, y)
    rho_s, p_s = spearmanr(x, y)
    ax.scatter(x, y, s=8, alpha=0.35, edgecolors="none", color="steelblue")
    # Linear fit overlay
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, np.polyval(coef, xs), "r-", linewidth=1.2, alpha=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (n={len(x)})\n"
                 f"Pearson r={rho_p:+.3f}  Spearman ρ={rho_s:+.3f}",
                 fontsize=10)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", type=str, default="pythia-70m")
    ap.add_argument("--timestamp", type=str, default=None,
                    help="Pin entropy_comparison files to this timestamp.")
    ap.add_argument("--levels-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    levels_path = (args.levels_file or
                   ROOT / "data" / args.preset / "feature_level_entropies.pt")
    levels = torch.load(levels_path, map_location="cpu", weights_only=False)
    cfg = levels.get("config", {})
    layer_keys = sorted(k for k in levels if k.startswith("layer_"))
    if not layer_keys:
        raise RuntimeError(f"No layer_* keys in {levels_path}")

    n_layers = len(layer_keys)
    fig, axes = plt.subplots(n_layers, 2, figsize=(11, 3.0 * n_layers))
    if n_layers == 1:
        axes = np.array([axes])

    summary_rows = []
    for row_i, key in enumerate(layer_keys):
        layer = int(key.split("_")[1])
        slot = levels[key]
        H_logit = slot["H_logit"]                      # [n_latent]
        H_vocab = slot["H_vocab"]                      # [n_latent]
        n_latent = H_logit.shape[0]

        comparison_file = find_entropy_comparison_file(
            args.preset, layer, args.timestamp)
        feat_mean = aggregate_feature_entropy(comparison_file)
        # Build aligned arrays
        H_feat = np.full(n_latent, np.nan, dtype=np.float32)
        for fid, m in feat_mean.items():
            if 0 <= fid < n_latent:
                H_feat[fid] = m

        # Panel 1: H_feat vs H_vocab
        make_scatter(axes[row_i, 0], H_vocab, H_feat,
                     title=f"Layer {layer} — H_feat vs H_vocab",
                     xlabel="H_vocab (bits, vocab-side localization)")
        # Panel 2: H_feat vs H_logit
        make_scatter(axes[row_i, 1], H_logit, H_feat,
                     title=f"Layer {layer} — H_feat vs H_logit",
                     xlabel="H_logit (bits, output-side localization)")

        # Summary stats
        m_vocab = np.isfinite(H_vocab) & np.isfinite(H_feat)
        m_logit = np.isfinite(H_logit) & np.isfinite(H_feat)
        if m_vocab.sum() >= 5:
            r_v, p_v = pearsonr(H_vocab[m_vocab], H_feat[m_vocab])
            s_v, _ = spearmanr(H_vocab[m_vocab], H_feat[m_vocab])
        else:
            r_v = p_v = s_v = float("nan")
        if m_logit.sum() >= 5:
            r_l, p_l = pearsonr(H_logit[m_logit], H_feat[m_logit])
            s_l, _ = spearmanr(H_logit[m_logit], H_feat[m_logit])
        else:
            r_l = p_l = s_l = float("nan")
        summary_rows.append({
            "layer": layer,
            "n_features_overlap_vocab": int(m_vocab.sum()),
            "n_features_overlap_logit": int(m_logit.sum()),
            "pearson_H_feat_vs_H_vocab": float(r_v),
            "spearman_H_feat_vs_H_vocab": float(s_v),
            "pearson_p_H_vocab": float(p_v),
            "pearson_H_feat_vs_H_logit": float(r_l),
            "spearman_H_feat_vs_H_logit": float(s_l),
            "pearson_p_H_logit": float(p_l),
            "mean_H_feat": float(np.nanmean(H_feat)),
            "mean_H_vocab": float(np.nanmean(H_vocab)),
            "mean_H_logit": float(np.nanmean(H_logit)),
        })

    fig.suptitle(f"Feature-intrinsic 'level' entropies vs influence entropy "
                 f"({cfg.get('preset', args.preset)}, threshold={cfg.get('threshold')})",
                 fontsize=13, y=1.005)
    fig.tight_layout()

    out_path = args.out or (ROOT / "plots" /
                            f"feature_level_vs_influence_entropy_{args.preset}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    # Save summary stats next to the plot
    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump({"config": cfg, "per_layer": summary_rows}, f, indent=2)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
