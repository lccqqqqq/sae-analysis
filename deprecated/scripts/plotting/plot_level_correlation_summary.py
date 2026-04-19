"""Summary figure for the H_vocab/H_logit vs H_feat trial.

Two-row layout:
  Top    — bar chart of correlation coefficients vs layer (Pearson + Spearman),
           one panel for H_vocab and one for H_logit, with significance stars.
  Bottom — three illustrative H_vocab-vs-H_feat scatter panels (one shallow,
           one transition, one deep) showing the regime change visually.

Reads everything from the JSON written by plot_feature_level_vs_entropy.py
plus the entropy_comparison_*.pt files (for the scatter inset data).

Usage:
    python scripts/plotting/plot_level_correlation_summary.py \
        --preset pythia-70m --timestamp 20260414_053350
"""

from __future__ import annotations

import argparse
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
    if timestamp:
        p = ROOT / "data" / f"entropy_comparison_resid_out_layer{layer}_{timestamp}.pt"
        if p.exists():
            return p
        p2 = ROOT / "data" / preset / timestamp / f"entropy_comparison_resid_out_layer{layer}.pt"
        if p2.exists():
            return p2
    cand = list((ROOT / "data").glob(f"entropy_comparison_resid_out_layer{layer}_*.pt"))
    cand += list((ROOT / "data" / preset).glob(f"*/entropy_comparison_resid_out_layer{layer}.pt"))
    if not cand:
        raise FileNotFoundError(f"No entropy_comparison file for layer {layer}")
    return max(cand, key=lambda p: p.stat().st_mtime)


def aggregate_feature_entropy(file_path: Path) -> dict[int, float]:
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    bucket: dict[int, list[float]] = {}
    for br in data["batch_results"]:
        for fid, ent in (br.get("feature_entropies") or {}).items():
            bucket.setdefault(int(fid), []).append(float(ent))
    return {fid: float(np.mean(v)) for fid, v in bucket.items()}


def sig_marker(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-10: return "***"
    if p < 1e-3:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", type=str, default="pythia-70m")
    ap.add_argument("--timestamp", type=str, default="20260414_053350")
    ap.add_argument("--levels-file", type=Path, default=None)
    ap.add_argument("--summary-json", type=Path, default=None)
    ap.add_argument("--scatter-layers", type=int, nargs=3, default=[0, 2, 5],
                    help="Three representative layers for the bottom row.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    levels_path = (args.levels_file or
                   ROOT / "data" / args.preset / "feature_level_entropies.pt")
    levels = torch.load(levels_path, map_location="cpu", weights_only=False)
    summary_path = (args.summary_json or
                    ROOT / "plots" /
                    f"feature_level_vs_influence_entropy_{args.preset}.json")
    with open(summary_path) as f:
        summary = json.load(f)["per_layer"]

    layers = [row["layer"] for row in summary]
    pear_v = [row["pearson_H_feat_vs_H_vocab"] for row in summary]
    spe_v  = [row["spearman_H_feat_vs_H_vocab"] for row in summary]
    pp_v   = [row["pearson_p_H_vocab"] for row in summary]
    pear_l = [row["pearson_H_feat_vs_H_logit"] for row in summary]
    spe_l  = [row["spearman_H_feat_vs_H_logit"] for row in summary]
    pp_l   = [row["pearson_p_H_logit"] for row in summary]
    n_v    = [row["n_features_overlap_vocab"] for row in summary]
    n_l    = [row["n_features_overlap_logit"] for row in summary]

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.42, wspace=0.30)

    # ---- Top row: correlation vs layer ----
    ax_v = fig.add_subplot(gs[0, :2])
    x = np.arange(len(layers))
    w = 0.36
    bars1 = ax_v.bar(x - w/2, pear_v, w, label="Pearson r",
                     color="steelblue", edgecolor="black", linewidth=0.5)
    bars2 = ax_v.bar(x + w/2, spe_v, w, label="Spearman ρ",
                     color="tomato", edgecolor="black", linewidth=0.5)
    for xi, p_pearson, p_val, n in zip(x, pear_v, pp_v, n_v):
        ax_v.text(xi - w/2, p_pearson + (0.02 if p_pearson >= 0 else -0.05),
                  sig_marker(p_val), ha="center", fontsize=9)
        ax_v.text(xi, -0.10, f"n={n}", ha="center", fontsize=8, color="gray")
    ax_v.axhline(0, color="black", linewidth=0.6)
    ax_v.set_xticks(x); ax_v.set_xticklabels([f"L{l}" for l in layers])
    ax_v.set_ylabel("correlation with $H_\\text{feat}$")
    ax_v.set_title("$H_\\text{vocab}$ vs $H_\\text{feat}$ — correlation by layer",
                   fontsize=11, fontweight="bold")
    ax_v.set_ylim(-0.15, 0.75)
    ax_v.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax_v.grid(True, alpha=0.3, axis="y")

    ax_l = fig.add_subplot(gs[0, 2])
    ax_l.bar(x - w/2, pear_l, w, color="steelblue", edgecolor="black", linewidth=0.5)
    ax_l.bar(x + w/2, spe_l, w, color="tomato", edgecolor="black", linewidth=0.5)
    for xi, p_pearson, p_val in zip(x, pear_l, pp_l):
        ax_l.text(xi - w/2, p_pearson + (0.012 if p_pearson >= 0 else -0.025),
                  sig_marker(p_val), ha="center", fontsize=8)
    ax_l.axhline(0, color="black", linewidth=0.6)
    ax_l.set_xticks(x); ax_l.set_xticklabels([f"L{l}" for l in layers])
    ax_l.set_ylabel("correlation with $H_\\text{feat}$")
    ax_l.set_title("$H_\\text{logit}$ vs $H_\\text{feat}$\n(degenerate, see report)",
                   fontsize=10)
    ax_l.set_ylim(-0.15, 0.20)
    ax_l.grid(True, alpha=0.3, axis="y")

    # ---- Bottom row: three illustrative scatters of H_feat vs H_vocab ----
    captions = {0: "Layer 0 — strong positive",
                2: "Layer 2 — collapse begins",
                5: "Layer 5 — null"}

    for col, layer in enumerate(args.scatter_layers):
        ax = fig.add_subplot(gs[1, col])
        slot = levels[f"layer_{layer}"]
        H_vocab = slot["H_vocab"]
        H_logit = slot["H_logit"]
        n_latent = H_logit.shape[0]
        comp_file = find_entropy_comparison_file(args.preset, layer, args.timestamp)
        feat_mean = aggregate_feature_entropy(comp_file)
        H_feat = np.full(n_latent, np.nan, dtype=np.float32)
        for fid, m in feat_mean.items():
            if 0 <= fid < n_latent:
                H_feat[fid] = m
        mask = np.isfinite(H_vocab) & np.isfinite(H_feat)
        x_arr = H_vocab[mask]; y_arr = H_feat[mask]
        ax.scatter(x_arr, y_arr, s=10, alpha=0.32, edgecolors="none",
                   color="steelblue")
        if x_arr.size >= 5:
            r, p = pearsonr(x_arr, y_arr)
            rho, _ = spearmanr(x_arr, y_arr)
            xs = np.linspace(x_arr.min(), x_arr.max(), 50)
            coef = np.polyfit(x_arr, y_arr, 1)
            ax.plot(xs, np.polyval(coef, xs), "r-", linewidth=1.5, alpha=0.85)
            txt = f"r = {r:+.3f}\nρ = {rho:+.3f}\nn = {x_arr.size}"
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top",
                    fontsize=9, family="monospace",
                    bbox=dict(facecolor="white", edgecolor="lightgray",
                              boxstyle="round,pad=0.3"))
        ax.set_xlabel("$H_\\text{vocab}$ (bits)")
        ax.set_ylabel("$H_\\text{feat}$ (bits)")
        ax.set_title(captions.get(layer, f"Layer {layer}"), fontsize=10,
                     fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Feature-intrinsic vocabulary breadth predicts influence "
                 "entropy at shallow layers only\n"
                 f"(pythia-70m, threshold = 1.0, * p<0.05  ** p<1e-3  *** p<1e-10)",
                 fontsize=12, y=0.995)

    out_path = (args.out or
                ROOT / "plots" /
                f"feature_level_correlation_summary_{args.preset}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
