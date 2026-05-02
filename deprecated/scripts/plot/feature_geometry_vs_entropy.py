"""Visualise pairwise (|cosine|, |ΔH_pos|) and per-feature neighbour test.

Reads ``data/feature_geometry_vs_entropy/<TIMESTAMP>/`` and writes:

  figures/abscos_vs_dH_scatter.png
      Hex-bin density of all pairs (|cos| on x).
  figures/abscos_vs_dH_binned.png
      Mean |ΔH_pos| ±1 SEM in equal-count |cos| bins.
  figures/feature_vs_neighbour_H.png
      One point per feature: H_pos(a) on x, mean H_pos of its top-K
      |cos|-neighbours on y. y=x diagonal as the null. Spearman
      reported in the title. This is the strongest summary plot.
  figures/umap_decoder_colored_by_H.png
      UMAP (cosine metric) of the 298 decoder vectors, points coloured
      by H_pos. PCA fallback panel for orientation.
  figures/h_quantile_geometry.png
      Group features into low/mid/high H_pos terciles. For each pair
      of groups, plot the within-group |cos| distribution vs the
      between-group distribution. If geometry tracks entropy, the
      within-group distribution is shifted right.

Usage:
    python scripts/plot/feature_geometry_vs_entropy.py \\
        --expt-dir data/feature_geometry_vs_entropy/latest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

def plot_scatter_hexbin(pairs: pd.DataFrame, out: Path):
    x = pairs["abs_cosine"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    hb = ax.hexbin(x, pairs["dH"], gridsize=60,
                   cmap="viridis", bins="log", mincnt=1)
    plt.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_xlabel("|decoder cosine similarity|", fontsize=11)
    ax.set_ylabel("|ΔH_pos|  (bits)", fontsize=11)
    ax.set_title(f"All pairs  (n={len(pairs):,})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {out}")


def _equal_count_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Return bin edges that put roughly equal counts in each bin."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def plot_binned(pairs: pd.DataFrame, manifest: dict, out: Path,
                n_bins: int = 24):
    edges = _equal_count_bins(pairs["abs_cosine"].to_numpy(), n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cos = pairs["abs_cosine"].to_numpy()
    dH = pairs["dH"].to_numpy()
    means = np.empty(n_bins)
    sems = np.empty(n_bins)
    counts = np.empty(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (cos > edges[i]) & (cos <= edges[i + 1])
        bin_dH = dH[mask]
        counts[i] = bin_dH.size
        means[i] = bin_dH.mean() if bin_dH.size else np.nan
        sems[i] = bin_dH.std(ddof=1) / np.sqrt(bin_dH.size) if bin_dH.size > 1 else 0.0

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(centres, means, yerr=sems, fmt="o-", color="#1f77b4",
                ecolor="#1f77b4", capsize=3, label="binned mean ±1 SEM")
    ax.axhline(dH.mean(), color="grey", linestyle="--", alpha=0.6,
               label=f"global mean |ΔH_pos|={dH.mean():.2f}")
    ax.set_xlabel("|decoder cosine similarity|  (equal-count bins)",
                  fontsize=11)
    ax.set_ylabel("mean |ΔH_pos|  (bits)", fontsize=11)
    sp_r = manifest.get("spearman_r")
    pe_r = manifest.get("pearson_r")
    p_perm = manifest.get("permutation_p_two_sided")
    z = manifest.get("permutation_z")
    ax.set_title(
        f"|ΔH_pos| vs |cosine|  "
        f"(Spearman={sp_r:+.3f}, Pearson={pe_r:+.3f}, "
        f"perm z={z:+.2f}, p={p_perm:.3f})",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {out}")


def plot_feature_vs_neighbour(nbr_df: pd.DataFrame, manifest: dict, out: Path):
    """Per-feature: H_pos(a) vs mean H_pos of its top-K |cos| neighbours."""
    K = manifest["per_feature_neighbour"]["k"]
    sp_r = manifest["per_feature_neighbour"]["spearman_r"]
    sp_p = manifest["per_feature_neighbour"]["spearman_p"]
    Hx = nbr_df["H_pos"].to_numpy()
    Hy = nbr_df[f"nbr{K}_H_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6.5))
    sc = ax.scatter(Hx, Hy, c=nbr_df[f"nbr{K}_abs_cos_mean"],
                    cmap="plasma", s=28, edgecolor="black",
                    linewidth=0.3, alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"mean |cos| of top-{K} nbrs", fontsize=10)
    lo = float(min(Hx.min(), Hy.min())) - 0.2
    hi = float(max(Hx.max(), Hy.max())) + 0.2
    ax.plot([lo, hi], [lo, hi], "--", color="grey", alpha=0.6,
            label="y = x  (perfect agreement)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("H_pos(a)  (bits)", fontsize=11)
    ax.set_ylabel(f"mean H_pos of top-{K} |cos| neighbours  (bits)",
                  fontsize=11)
    ax.set_title(
        f"Per-feature neighbour test  "
        f"Spearman={sp_r:+.3f}  p={sp_p:.1e}  (n={len(nbr_df)})",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {out}")


def plot_h_quantile_geometry(per_feat: pd.DataFrame, D: np.ndarray,
                             out: Path, n_groups: int = 3):
    """Group features by H_pos quantile; compare within- vs between-group
    |cos| distributions. If geometry tracks entropy, within-group has
    higher mean |cos|.
    """
    H = per_feat["H_pos_mean"].to_numpy()
    qs = np.linspace(0, 1, n_groups + 1)
    edges = np.quantile(H, qs)
    edges[-1] += 1e-9
    edges[0] -= 1e-9
    group = np.digitize(H, edges) - 1
    group = np.clip(group, 0, n_groups - 1)
    group_names = []
    for g in range(n_groups):
        sub = H[group == g]
        group_names.append(f"H ∈ [{sub.min():.2f},{sub.max():.2f}]  n={sub.size}")

    Dn = D / np.linalg.norm(D, axis=1, keepdims=True).clip(min=1e-12)
    cos = Dn @ Dn.T
    abs_cos = np.abs(cos)
    np.fill_diagonal(abs_cos, np.nan)

    # Per (i,j) within-group / between-group masks; plot violin.
    same_group = (group[:, None] == group[None, :])
    iu, ju = np.triu_indices(len(H), k=1)
    pair_same = same_group[iu, ju]
    pair_cos = abs_cos[iu, ju]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left panel: within vs between violins.
    parts = axes[0].violinplot(
        [pair_cos[pair_same], pair_cos[~pair_same]],
        positions=[0, 1], widths=0.7, showmeans=True, showmedians=True,
    )
    for body, c in zip(parts["bodies"], ["#d62728", "#1f77b4"]):
        body.set_facecolor(c)
        body.set_alpha(0.5)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(
        [f"same H-band  (n={pair_same.sum():,})",
         f"different H-band  (n={(~pair_same).sum():,})"],
        fontsize=9,
    )
    axes[0].set_ylabel("|decoder cosine|", fontsize=11)
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(pair_cos[pair_same], pair_cos[~pair_same],
                        alternative="greater")
    mean_within = pair_cos[pair_same].mean()
    mean_between = pair_cos[~pair_same].mean()
    axes[0].set_title(
        f"Within- vs between-H-band |cos|\n"
        f"means: {mean_within:.4f} vs {mean_between:.4f}, MW p={p:.2e}",
        fontsize=11, fontweight="bold",
    )
    axes[0].grid(True, axis="y", alpha=0.3)

    # Right panel: per-band mean |cos| to each other band (heatmap).
    M = np.zeros((n_groups, n_groups))
    for a in range(n_groups):
        for b in range(n_groups):
            mask_a = group == a
            mask_b = group == b
            block = abs_cos[np.ix_(mask_a, mask_b)]
            M[a, b] = np.nanmean(block)
    im = axes[1].imshow(M, cmap="plasma", origin="lower")
    plt.colorbar(im, ax=axes[1], label="mean |cos|")
    axes[1].set_xticks(range(n_groups))
    axes[1].set_yticks(range(n_groups))
    axes[1].set_xticklabels([f"H{g}" for g in range(n_groups)])
    axes[1].set_yticklabels([f"H{g}" for g in range(n_groups)])
    for a in range(n_groups):
        for b in range(n_groups):
            axes[1].text(b, a, f"{M[a, b]:.4f}", ha="center", va="center",
                         color="white", fontsize=10, fontweight="bold")
    band_lines = "  ·  ".join(
        f"H{g}: H∈[{H[group==g].min():.2f},{H[group==g].max():.2f}]"
        for g in range(n_groups)
    )
    axes[1].set_title(f"Block-mean |cos| between H-bands\n{band_lines}",
                      fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Decoder geometry by H_pos quantile band  ({n_groups} bands)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {out}")


def plot_umap_pca(per_feat: pd.DataFrame, D: np.ndarray, out: Path):
    H = per_feat["H_pos_mean"].to_numpy()

    # PCA via SVD on L2-normalised D (each row a feature).
    Dn = D / np.linalg.norm(D, axis=1, keepdims=True).clip(min=1e-12)
    Dc = Dn - Dn.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Dc, full_matrices=False)
    pca2 = U[:, :2] * S[:2]

    # UMAP (cosine metric). Falls back gracefully if umap-learn missing.
    umap_xy = None
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                            random_state=0)
        umap_xy = reducer.fit_transform(Dn)
    except Exception as e:
        print(f"[WARN] UMAP unavailable ({type(e).__name__}: {e}); "
              f"PCA panel only.")

    n_panels = 2 if umap_xy is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    vmin, vmax = float(np.percentile(H, 2)), float(np.percentile(H, 98))
    sc = None
    for ax, xy, title in zip(
        axes,
        ([umap_xy, pca2] if umap_xy is not None else [pca2]),
        (["UMAP (cosine)", "PCA (top-2)"] if umap_xy is not None else ["PCA (top-2)"]),
    ):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=H, cmap="viridis",
                        vmin=vmin, vmax=vmax, s=28, edgecolor="black",
                        linewidth=0.3, alpha=0.9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("H_pos (bits)", fontsize=10)
    fig.suptitle(
        f"Decoder geometry coloured by H_pos  (N={len(H)})",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {out}")
    return pca2, umap_xy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expt-dir", type=Path, required=True)
    args = ap.parse_args()

    expt_dir = args.expt_dir.resolve()
    figdir = expt_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((expt_dir / "manifest.json").read_text())
    per_feat = pd.read_csv(expt_dir / "per_feature.csv")
    pairs = pd.read_csv(expt_dir / "pairwise.csv")
    nbr_df = pd.read_csv(expt_dir / "feature_neighbour_summary.csv")
    D = torch.load(expt_dir / "decoder_vectors.pt").cpu().numpy()

    plot_scatter_hexbin(pairs, figdir / "abscos_vs_dH_scatter.png")
    plot_binned(pairs, manifest, figdir / "abscos_vs_dH_binned.png")
    plot_feature_vs_neighbour(nbr_df, manifest, figdir / "feature_vs_neighbour_H.png")
    plot_umap_pca(per_feat, D, figdir / "umap_decoder_colored_by_H.png")
    plot_h_quantile_geometry(per_feat, D, figdir / "h_quantile_geometry.png")
    print(f"[INFO] all figures under {figdir}")


if __name__ == "__main__":
    main()
