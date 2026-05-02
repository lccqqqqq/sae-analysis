"""Interactive 3D plot: PCA(2) of decoder vectors as XY, H_pos as Z.

For 6 anchor features and their top-K |cos| neighbours over the full
SAE, plot one marker per feature at (x, y) = its PCA-of-decoder
coordinates and z = H_pos. Vertical error bars on z show the
per-event spread (IQR by default, or min/max).

Hover tooltip exposes feature_id, autointerp description, |cos| to
its anchor, and the H_pos summary stats.

Optional: also overlay the 298-feature background as small grey dots.

Usage:
    python scripts/plot/feature_geometry_clusters_3d.py \\
        --cluster-info /tmp/cluster_anchors_nbrs.pt \\
        --entropy-dir data/cherry_picked_feature_entropy/latest_neuronpedia_anchor6_topk10 \\
        --geom-dir data/feature_geometry_vs_entropy/latest \\
        --output data/feature_geometry_vs_entropy/latest/figures/clusters3d.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "analysis"))


def load_events(entropy_dir: Path, ctx_len: int = 128) -> pd.DataFrame:
    frames = []
    for tier in ["token", "phrase", "concept", "abstract"]:
        p = entropy_dir / f"ctx{ctx_len:03d}_{tier}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"no ctx{ctx_len:03d}_*.csv under {entropy_dir}")
    df = pd.concat(frames, ignore_index=True)
    return df[df["status"] == "ok"].copy()


def per_feature_stats(events: pd.DataFrame) -> pd.DataFrame:
    g = events.groupby("feature_id")["H_pos_bits"]
    return pd.DataFrame({
        "feature_id": g.size().index,
        "n_events": g.size().values,
        "H_mean": g.mean().values,
        "H_median": g.median().values,
        "H_std": g.std().values,
        "H_min": g.min().values,
        "H_max": g.max().values,
        "H_q25": g.quantile(0.25).values,
        "H_q75": g.quantile(0.75).values,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster-info", type=Path, required=True,
                    help="torch.save dict with anchors / nbr_per_anchor.")
    ap.add_argument("--entropy-dir", type=Path, required=True,
                    help="Per-event entropy CSVs for the 66 features.")
    ap.add_argument("--geom-dir", type=Path, required=True,
                    help="latest feature_geometry_vs_entropy/ dir, used "
                         "to fetch the 298-feature background.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ctx-len", type=int, default=128)
    ap.add_argument("--whisker", choices=["iqr", "minmax"], default="iqr")
    ap.add_argument("--projection", choices=["pca", "tsne", "umap"],
                    default="tsne",
                    help="2D projection of decoder vectors for the XY plane.")
    ap.add_argument("--tsne-perplexity", type=float, default=10.0,
                    help="t-SNE perplexity (low for small N=66; default 10).")
    ap.add_argument("--with-background", action="store_true", default=False,
                    help="Overlay 298-feature backdrop (off by default).")
    ap.add_argument("--no-background", dest="with_background",
                    action="store_false")
    ap.add_argument("--draw-edges", action="store_true", default=True,
                    help="Draw thin lines from each neighbour to its anchor.")
    ap.add_argument("--no-edges", dest="draw_edges", action="store_false")
    args = ap.parse_args()

    info = torch.load(args.cluster_info, weights_only=False)
    anchors = list(info["anchors"])
    nbr_per_anchor = info["nbr_per_anchor"]
    needed_fids = sorted(info["all_needed_fids"])
    fid_to_idx = {fid: i for i, fid in enumerate(needed_fids)}

    # --- per-event H_pos for the 66 cluster features -----------------------
    events = load_events(args.entropy_dir, ctx_len=args.ctx_len)
    events = events[events["feature_id"].isin(needed_fids)]
    stats = per_feature_stats(events)
    desc_map = {int(r["feature_id"]): str(r["description"])
                for _, r in events.drop_duplicates("feature_id").iterrows()}

    # --- decoder vectors for those 66 features -----------------------------
    from presets import get_preset
    from sae_adapters import load_sae
    preset = get_preset("gemma-2-2b")
    bundle = load_sae(preset, layer_idx=12, device="cpu")
    D_full = bundle.dec_w.T.contiguous()
    D = D_full[needed_fids].numpy()
    Dn = D / np.linalg.norm(D, axis=1, keepdims=True).clip(min=1e-12)

    # 2D projection of the 66 decoder vectors.
    Dc = Dn - Dn.mean(axis=0, keepdims=True)
    if args.projection == "pca":
        U, S, _ = np.linalg.svd(Dc, full_matrices=False)
        xy2 = U[:, :2] * S[:2]
        proj_label = "PCA"
    elif args.projection == "tsne":
        from sklearn.manifold import TSNE
        # 1 - |cos| as distance keeps antipodal pairs together (matches
        # the |cos| analysis upstream).
        sim = np.abs(Dn @ Dn.T)
        np.fill_diagonal(sim, 1.0)
        dist = np.clip(1.0 - sim, 0.0, None)
        tsne = TSNE(n_components=2, metric="precomputed",
                    perplexity=args.tsne_perplexity, init="random",
                    random_state=0, max_iter=2000)
        xy2 = tsne.fit_transform(dist)
        proj_label = f"t-SNE (perp={args.tsne_perplexity:g})"
    elif args.projection == "umap":
        import umap
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.15, metric="cosine",
                            random_state=0)
        xy2 = reducer.fit_transform(Dn)
        proj_label = "UMAP"
    fid_to_xy = {int(fid): (float(xy2[i, 0]), float(xy2[i, 1]))
                 for i, fid in enumerate(needed_fids)}

    # In-cluster |cos| matrix (66 × 66): per-feature top-3 nearest within
    # this 66-feature universe, used to enrich hover tooltips with the
    # autointerp of each feature's closest neighbours.
    sim_in = np.abs(Dn @ Dn.T)
    np.fill_diagonal(sim_in, -np.inf)
    top3_in = np.argsort(-sim_in, axis=1)[:, :3]
    in_cluster_top3 = {}
    for i, fid in enumerate(needed_fids):
        in_cluster_top3[int(fid)] = [
            (int(needed_fids[j]), float(sim_in[i, j])) for j in top3_in[i]
        ]

    # --- assign each fid to a primary anchor cluster -----------------------
    primary = {a: a for a in anchors}
    for a, nbrs in nbr_per_anchor.items():
        for fid, _ in nbrs:
            primary.setdefault(int(fid), int(a))
    cos_to_anchor = {int(a): {int(a): 1.0} for a in anchors}
    for a, nbrs in nbr_per_anchor.items():
        for fid, c in nbrs:
            cos_to_anchor.setdefault(int(a), {})[int(fid)] = float(c)

    # --- optional background: 298-feature decoder vectors --------------
    # Only meaningful for PCA (a linear projection we can extend to new
    # points). For t-SNE/UMAP the background is dropped because those
    # are non-linear maps that can't be re-applied to held-out points.
    bg_x = bg_y = bg_z = bg_text = None
    if args.with_background and args.projection == "pca":
        per_feat_298 = pd.read_csv(args.geom_dir / "per_feature.csv")
        bg_fids = per_feat_298["feature_id"].astype(int).tolist()
        D_bg = D_full[bg_fids].numpy()
        Dn_bg = D_bg / np.linalg.norm(D_bg, axis=1, keepdims=True).clip(min=1e-12)
        _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
        proj = (Dn_bg - Dn.mean(axis=0, keepdims=True)) @ Vt[:2].T
        bg_x = proj[:, 0]
        bg_y = proj[:, 1]
        bg_z = per_feat_298["H_pos_mean"].to_numpy()
        bg_text = [f"feature {f}<br>H_pos≈{h:.2f}" for f, h in
                   zip(bg_fids, bg_z)]

    # --- render with plotly ------------------------------------------------
    import plotly.graph_objects as go

    fig = go.Figure()

    if args.with_background and bg_x is not None:
        fig.add_trace(go.Scatter3d(
            x=bg_x, y=bg_y, z=bg_z,
            mode="markers",
            marker=dict(size=2.5, color="lightgrey", opacity=0.35),
            name="298-feature background",
            text=bg_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ))

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
               "#ff7f0e", "#17becf"]

    def _build_hover(fid, is_anchor, anchor, zmed, row):
        cos_val = cos_to_anchor[anchor].get(fid, 1.0)
        nbr_lines = []
        for nb_fid, nb_cos in in_cluster_top3.get(fid, []):
            nb_desc = desc_map.get(nb_fid, "")[:120]
            nbr_lines.append(
                f"&nbsp;&nbsp;• f{nb_fid} (|cos|={nb_cos:.3f}): {nb_desc}"
            )
        nbr_block = "<br>".join(nbr_lines) if nbr_lines else ""
        return (
            f"<b>feature {fid}{' [ANCHOR]' if is_anchor else ''}</b><br>"
            f"|cos| to anchor {anchor}: {cos_val:.3f}<br>"
            f"H_pos: median={zmed:.2f}, "
            f"IQR=[{float(row['H_q25'].values[0]):.2f},"
            f"{float(row['H_q75'].values[0]):.2f}], "
            f"range=[{float(row['H_min'].values[0]):.2f},"
            f"{float(row['H_max'].values[0]):.2f}], "
            f"n={int(row['n_events'].values[0])}<br>"
            f"<i>{desc_map.get(fid, '')[:200]}</i><br>"
            f"<b>top-3 in-cluster nbrs:</b><br>{nbr_block}"
        )

    def _zrange(row):
        zmed = float(row["H_median"].values[0])
        if args.whisker == "iqr":
            return (zmed,
                    zmed - float(row["H_q25"].values[0]),
                    float(row["H_q75"].values[0]) - zmed)
        return (zmed,
                zmed - float(row["H_min"].values[0]),
                float(row["H_max"].values[0]) - zmed)

    # Layer 1: connecting lines anchor -> each in-cluster neighbour.
    # One Scatter3d trace per anchor, with NaN-separated segments.
    if args.draw_edges:
        for ai, anchor in enumerate(anchors):
            col = palette[ai % len(palette)]
            if anchor not in fid_to_xy:
                continue
            ax_, ay_ = fid_to_xy[anchor]
            anchor_row = stats[stats["feature_id"] == anchor]
            if anchor_row.empty:
                continue
            az_ = float(anchor_row["H_median"].values[0])
            xs = []; ys = []; zs = []
            for fid, _ in nbr_per_anchor[anchor]:
                fid = int(fid)
                if primary.get(fid) != anchor:
                    continue
                if fid not in fid_to_xy:
                    continue
                nrow = stats[stats["feature_id"] == fid]
                if nrow.empty:
                    continue
                nx, ny = fid_to_xy[fid]
                nz = float(nrow["H_median"].values[0])
                xs.extend([ax_, nx, None])
                ys.extend([ay_, ny, None])
                zs.extend([az_, nz, None])
            if xs:
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color=col, width=2),
                    opacity=0.35,
                    name=f"edges {anchor}",
                    hoverinfo="skip", showlegend=False,
                ))

    # Layer 2: non-anchor neighbours.
    for ai, anchor in enumerate(anchors):
        col = palette[ai % len(palette)]
        nbr_fids = [int(f) for f, _ in nbr_per_anchor[anchor]]
        nbr_fids = [f for f in nbr_fids if primary.get(f) == anchor]
        xs, ys, zs, zlo, zhi, txt = [], [], [], [], [], []
        for fid in nbr_fids:
            if fid not in fid_to_xy: continue
            row = stats[stats["feature_id"] == fid]
            if row.empty: continue
            x, y = fid_to_xy[fid]
            zmed, lo, hi = _zrange(row)
            xs.append(x); ys.append(y); zs.append(zmed)
            zlo.append(lo); zhi.append(hi)
            txt.append(_build_hover(fid, False, anchor, zmed, row))
        if xs:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers",
                marker=dict(size=5, color=col, opacity=0.9,
                            symbol="circle",
                            line=dict(color="black", width=0.5)),
                error_z=dict(type="data", symmetric=False,
                             array=zhi, arrayminus=zlo,
                             color=col, thickness=1.5, width=3),
                name=f"anchor {anchor} nbrs",
                text=txt, hovertemplate="%{text}<extra></extra>",
                legendgroup=f"a{anchor}",
            ))

    # Layer 3 (top): anchors themselves — large diamonds, drawn last.
    for ai, anchor in enumerate(anchors):
        col = palette[ai % len(palette)]
        if anchor not in fid_to_xy: continue
        row = stats[stats["feature_id"] == anchor]
        if row.empty: continue
        x, y = fid_to_xy[anchor]
        zmed, lo, hi = _zrange(row)
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[zmed], mode="markers",
            marker=dict(size=8, color=col, opacity=1.0,
                        symbol="diamond",
                        line=dict(color="black", width=1.5)),
            error_z=dict(type="data", symmetric=False,
                         array=[hi], arrayminus=[lo],
                         color=col, thickness=3, width=6),
            name=f"anchor {anchor}",
            text=[_build_hover(anchor, True, anchor, zmed, row)],
            hovertemplate="%{text}<extra></extra>",
            legendgroup=f"a{anchor}",
        ))

    whisker_label = ("IQR (Q25–Q75)" if args.whisker == "iqr"
                     else "min–max")
    fig.update_layout(
        title=dict(
            text=f"Decoder {proj_label} × H_pos for 6 anchors + top-{info['k']} "
                 f"|cos| neighbours<br><sub>z error bars = per-event H_pos "
                 f"{whisker_label} (layer 12, gemma-scope-res-16k, "
                 f"ctx_len={args.ctx_len})</sub>",
            x=0.5, xanchor="center",
        ),
        scene=dict(
            xaxis=dict(title=f"{proj_label} dim 1"),
            yaxis=dict(title=f"{proj_label} dim 2"),
            zaxis=dict(title="H_pos (bits)"),
        ),
        legend=dict(itemsizing="constant", x=0.02, y=0.98),
        width=1100, height=800,
        template="plotly_white",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()
