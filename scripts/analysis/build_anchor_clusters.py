"""Build the anchor-clusters .pt blob consumed by feature_geometry_clusters_3d.py.

For each user-supplied *anchor feature*, this script searches the **full** SAE
decoder (16k features at gemma-2-2b L12 / 12-gemmascope-res-16k) for the
top-K |cos|-neighbours of the anchor's decoder vector, and saves the result
in the small dictionary format expected by the 3D viewer.

This is the search-over-all-16k step that was originally done ad-hoc in an
interactive session; committing it makes the whole experiment-3 pipeline
reproducible end-to-end.

Output (one ``torch.save`` blob)::

    {
        "anchors":         [a1, a2, ...],
        "k":               K,
        "nbr_per_anchor":  { a: [(fid, |cos|), ...] for a in anchors },
        "all_needed_fids": sorted union of anchors + all neighbours,
        "already_fids":    intersection with --per-feature CSV
                           (= 298-set features already with H_pos measured),
        "missing_fids":    all_needed_fids minus already_fids
                           (= features that still need H_pos measuring),
    }

Cosine convention: **absolute** cosine. This matches the upstream
``feature_geometry_vs_entropy.py`` which concluded that ``|cos|`` (not signed
cos) is the predictor that tracks H_pos — antipodal pairs share H_pos.

Usage::

    python scripts/analysis/build_anchor_clusters.py \\
        --anchors 608 4248 10360 6240 6900 13603 \\
        --k 10 \\
        --per-feature data/feature_geometry_vs_entropy/<TIMESTAMP>/per_feature.csv \\
        --output /tmp/cluster_anchors_nbrs.pt

The ``--per-feature`` arg is optional; if omitted, ``already_fids`` is empty
and ``missing_fids == all_needed_fids`` (which is the right state for a
fresh anchor set whose H_pos hasn't been measured yet).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from presets import get_preset
from sae_adapters import load_sae


def topk_abscos_neighbours(
    Dn: torch.Tensor,
    anchor: int,
    k: int,
) -> list[tuple[int, float]]:
    """Return the top-K |cos| neighbours of ``anchor`` over ``Dn`` (rows L2-normed)."""
    sim = (Dn @ Dn[anchor]).abs()
    sim[anchor] = -float("inf")
    vals, idx = sim.topk(k)
    return [(int(f), float(v)) for f, v in zip(idx.tolist(), vals.tolist())]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", type=int, nargs="+", required=True,
                    help="Anchor feature ids (within the SAE).")
    ap.add_argument("--k", type=int, default=10,
                    help="Number of |cos|-neighbours per anchor (default 10).")
    ap.add_argument("--preset", default="gemma-2-2b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--per-feature", type=Path, default=None,
                    help="Path to per_feature.csv from a feature_geometry_vs_entropy "
                         "run. Used to split needed_fids into already_fids vs "
                         "missing_fids. Optional.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Where to write the cluster-info .pt blob.")
    args = ap.parse_args()

    preset = get_preset(args.preset)
    bundle = load_sae(preset, layer_idx=args.layer, device="cpu")
    D = bundle.dec_w.T.contiguous()                            # [n_latent, d_model]
    Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)

    nbr_per_anchor = {
        a: topk_abscos_neighbours(Dn, a, args.k) for a in args.anchors
    }

    needed = set(args.anchors)
    for nbrs in nbr_per_anchor.values():
        needed.update(fid for fid, _ in nbrs)
    all_needed_fids = sorted(needed)

    if args.per_feature is not None:
        already = set(pd.read_csv(args.per_feature)["feature_id"].astype(int).tolist())
        already_fids = sorted(needed & already)
        missing_fids = sorted(needed - already)
    else:
        already_fids = []
        missing_fids = all_needed_fids

    blob = {
        "anchors":         list(args.anchors),
        "k":               args.k,
        "nbr_per_anchor":  nbr_per_anchor,
        "all_needed_fids": all_needed_fids,
        "missing_fids":    missing_fids,
        "already_fids":    already_fids,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.output)
    print(f"[INFO] wrote {args.output}")
    print(f"        anchors      = {blob['anchors']}")
    print(f"        k            = {blob['k']}")
    print(f"        |needed|     = {len(all_needed_fids)}")
    print(f"        |already|    = {len(already_fids)}")
    print(f"        |missing|    = {len(missing_fids)}")


if __name__ == "__main__":
    main()
