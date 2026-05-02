"""Test whether SAE feature decoder geometry tracks H_pos.

Hypothesis (informal): two features close in decoder cosine space have
similar H_pos. Falsifiable by computing all pairwise (cosine, |ΔH_pos|)
across the 298-feature dataset and looking for a negative slope (closer
↔ smaller entropy gap).

This script does only data computation. Plotting is in
``scripts/plot/feature_geometry_vs_entropy.py``.

Outputs (under ``data/feature_geometry_vs_entropy/<TIMESTAMP>/``):

  per_feature.csv
      feature_id, tier, description, n_events, H_pos_mean, H_pos_std

  decoder_vectors.pt
      torch tensor [N, d_model], one decoder row per feature, in the
      order of per_feature.csv (column ``row_idx``).

  cosine_matrix.pt
      torch tensor [N, N], full cosine-similarity matrix of those rows.

  pairwise.csv
      All N*(N-1)/2 unordered pairs:
      fid_a, fid_b, cosine, dH (|H_pos_a - H_pos_b|), H_a, H_b, tier_a, tier_b

  neuronpedia_xcheck.csv
      Cross-validation of our cosine values against Neuronpedia's
      precomputed topkCosSimValues for the in-set neighbour overlap.
      Columns: fid, neighbour_fid, cosine_local, cosine_neuronpedia, abs_diff.

  manifest.json
      preset / layer / sae_id, source ctx_len, source experiment dir,
      timestamp, n_features, n_pairs, summary statistics.

Usage:
    python scripts/analysis/feature_geometry_vs_entropy.py \\
        --source-dir data/cherry_picked_feature_entropy/latest_neuronpedia_full \\
        --ctx-len 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "analysis"))

from presets import get_preset      # noqa: E402
from sae_adapters import load_sae   # noqa: E402

TIER_ORDER = ["token", "phrase", "concept", "abstract"]


def aggregate_h_pos(source_dir: Path, ctx_len: int) -> pd.DataFrame:
    frames = []
    for tier in TIER_ORDER:
        p = source_dir / f"ctx{ctx_len:03d}_{tier}.csv"
        if not p.exists():
            print(f"[WARN] missing {p}")
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"no ctx{ctx_len:03d}_*.csv under {source_dir}")
    df = pd.concat(frames, ignore_index=True)
    ok = df[df["status"] == "ok"]
    agg = (ok.groupby(["feature_id", "tier", "description"], observed=True)
             .agg(n_events=("H_pos_bits", "size"),
                  H_pos_mean=("H_pos_bits", "mean"),
                  H_pos_std=("H_pos_bits", "std"))
             .reset_index()
             .sort_values("feature_id").reset_index(drop=True))
    return agg


def neuronpedia_xcheck(
    cache_root: Path, model_id: str, sae_id: str,
    fids: list[int], cos_matrix: np.ndarray, fid_to_row: dict[int, int],
) -> pd.DataFrame:
    """Compare local cosine values against Neuronpedia's precomputed
    topkCosSimValues, for in-set neighbour overlaps only.
    """
    rows = []
    cache_dir = cache_root / model_id / sae_id
    for fid in fids:
        path = cache_dir / f"{fid}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        idxs = d.get("topkCosSimIndices") or []
        vals = d.get("topkCosSimValues") or []
        for nb_fid, nb_cos in zip(idxs, vals):
            if nb_fid == fid:
                continue
            if int(nb_fid) not in fid_to_row:
                continue
            local = float(cos_matrix[fid_to_row[fid], fid_to_row[int(nb_fid)]])
            rows.append({
                "fid": int(fid),
                "neighbour_fid": int(nb_fid),
                "cosine_local": local,
                "cosine_neuronpedia": float(nb_cos),
                "abs_diff": abs(local - float(nb_cos)),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, required=True,
                    help="Experiment dir with ctx<N>_<tier>.csv files.")
    ap.add_argument("--ctx-len", type=int, default=128,
                    help="Which ctx_len to aggregate (default 128).")
    ap.add_argument("--output-root", type=Path,
                    default=Path("data/feature_geometry_vs_entropy"))
    ap.add_argument("--cache-root", type=Path,
                    default=Path("data/neuronpedia_cache"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    source_dir = args.source_dir.resolve()
    manifest_src = json.loads((source_dir / "manifest.json").read_text())
    preset_name = manifest_src["preset"]
    layer = int(manifest_src["layer"])
    sae_id = manifest_src["sae_id"]
    np_model_id = sae_id.split("-")[0]  # not used — Neuronpedia uses the
    # gemma-2-2b model id, sae_id with the "12-gemmascope-res-16k" form

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: aggregate per-feature H_pos.
    per_feat = aggregate_h_pos(source_dir, args.ctx_len)
    per_feat["row_idx"] = np.arange(len(per_feat))
    per_feat.to_csv(out_dir / "per_feature.csv", index=False)
    n_feat = len(per_feat)
    print(f"[INFO] aggregated H_pos for {n_feat} features at ctx_len={args.ctx_len}")

    # Step 2: load SAE and extract decoder rows.
    preset = get_preset(preset_name)
    print(f"[INFO] loading SAE: {preset_name} layer {layer} ...")
    bundle = load_sae(preset, layer_idx=layer, device=args.device)
    # dec_w is [d_model, n_latent]; we want one row per feature -> [n_latent, d_model]
    D_full = bundle.dec_w.T.contiguous()    # [n_latent, d_model]
    fids = per_feat["feature_id"].astype(int).tolist()
    D = D_full[fids].clone()                # [n_feat, d_model]
    torch.save(D, out_dir / "decoder_vectors.pt")
    print(f"[INFO] decoder rows: {tuple(D.shape)}")

    # Step 3: cosine matrix.
    Dn = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos = (Dn @ Dn.T).cpu().numpy()
    torch.save(torch.from_numpy(cos), out_dir / "cosine_matrix.pt")

    # Step 4: cross-check against Neuronpedia top-25.
    fid_to_row = {fid: i for i, fid in enumerate(fids)}
    xcheck = neuronpedia_xcheck(
        args.cache_root.resolve(), "gemma-2-2b", sae_id,
        fids, cos, fid_to_row,
    )
    xcheck.to_csv(out_dir / "neuronpedia_xcheck.csv", index=False)
    if len(xcheck):
        max_diff = xcheck["abs_diff"].max()
        mean_diff = xcheck["abs_diff"].mean()
        print(f"[INFO] Neuronpedia xcheck: {len(xcheck)} in-set neighbour pairs, "
              f"mean |Δ|={mean_diff:.4f}, max |Δ|={max_diff:.4f}")
    else:
        print(f"[WARN] no in-set neighbour overlap with Neuronpedia top-25 "
              f"(expected: most features have 0 neighbours within our 298)")

    # Step 5: pairwise table.
    H = per_feat["H_pos_mean"].to_numpy()
    tiers = per_feat["tier"].to_numpy()
    iu, ju = np.triu_indices(n_feat, k=1)
    pairs = pd.DataFrame({
        "fid_a": [fids[i] for i in iu],
        "fid_b": [fids[j] for j in ju],
        "cosine": cos[iu, ju],
        "H_a": H[iu],
        "H_b": H[ju],
        "dH": np.abs(H[iu] - H[ju]),
        "tier_a": tiers[iu],
        "tier_b": tiers[ju],
    })
    pairs.to_csv(out_dir / "pairwise.csv", index=False)
    print(f"[INFO] pairwise rows: {len(pairs)}")

    # Add |cosine| column; the signed cosine analysis turns out to be
    # misleading because two features with antipodal decoder rows share
    # the same axis (cf. feature splitting / absorption, Chanin 2024)
    # and so end up with similar H_pos.
    pairs["abs_cosine"] = pairs["cosine"].abs()
    pairs.to_csv(out_dir / "pairwise.csv", index=False)

    # Step 6: correlation summary -- both signed cosine and |cosine|.
    from scipy.stats import spearmanr, pearsonr
    stats = {}
    for key, x in [("signed", pairs["cosine"].to_numpy()),
                   ("abs", pairs["abs_cosine"].to_numpy())]:
        sp = spearmanr(x, pairs["dH"])
        pe = pearsonr(x, pairs["dH"])
        rng = np.random.default_rng(0)
        n_perm = 200
        null_r = np.empty(n_perm)
        for k in range(n_perm):
            perm = rng.permutation(n_feat)
            Hp = H[perm]
            dHp = np.abs(Hp[iu] - Hp[ju])
            null_r[k] = spearmanr(x, dHp).statistic
        z = ((sp.statistic - null_r.mean()) / null_r.std()
             if null_r.std() > 0 else float("nan"))
        p_perm = float((np.abs(null_r) >= abs(sp.statistic)).mean())
        print(f"[INFO] [{key:>6} cosine] Spearman={sp.statistic:+.4f} "
              f"(p={sp.pvalue:.2e})  Pearson={pe.statistic:+.4f} "
              f"(p={pe.pvalue:.2e})  perm-z={z:+.2f}  p_perm={p_perm:.4f}")
        stats[key] = {
            "spearman_r": float(sp.statistic),
            "spearman_p": float(sp.pvalue),
            "pearson_r": float(pe.statistic),
            "pearson_p": float(pe.pvalue),
            "permutation_null_mean": float(null_r.mean()),
            "permutation_null_std": float(null_r.std()),
            "permutation_z": float(z),
            "permutation_p_two_sided": float(p_perm),
            "n_perm": int(n_perm),
        }
    # Backwards-compat scalars at top level (using |cosine|, the
    # diagnostic predictor): everything else lives under "stats".
    sp_r = stats["abs"]["spearman_r"]
    sp_p = stats["abs"]["spearman_p"]
    pe_r = stats["abs"]["pearson_r"]
    pe_p = stats["abs"]["pearson_p"]
    z = stats["abs"]["permutation_z"]
    null_mean = stats["abs"]["permutation_null_mean"]
    null_std = stats["abs"]["permutation_null_std"]
    p_perm = stats["abs"]["permutation_p_two_sided"]
    n_perm = stats["abs"]["n_perm"]

    # Step 6b: per-feature neighbour-H_pos diagnostic.
    # For each feature, take its top-K |cosine| neighbours within the
    # 298-set, compare H_pos(a) to mean H_pos of those neighbours.
    K = 10
    abs_cos = np.abs(cos)
    np.fill_diagonal(abs_cos, -np.inf)  # exclude self
    nbr_idx = np.argsort(-abs_cos, axis=1)[:, :K]
    nbr_H_mean = H[nbr_idx].mean(axis=1)
    nbr_df = pd.DataFrame({
        "feature_id": fids,
        "tier": tiers,
        "H_pos": H,
        f"nbr{K}_H_mean": nbr_H_mean,
        f"nbr{K}_abs_cos_mean": np.sort(abs_cos, axis=1)[:, -K:].mean(axis=1),
    })
    nbr_df.to_csv(out_dir / "feature_neighbour_summary.csv", index=False)
    nbr_sp = spearmanr(H, nbr_H_mean)
    print(f"[INFO] per-feature: Spearman(H_pos, mean H_pos of top-{K} |cos| nbrs) "
          f"= {nbr_sp.statistic:+.3f}  p={nbr_sp.pvalue:.2e}  (n={n_feat})")

    # Manifest.
    manifest = {
        "experiment": "feature_geometry_vs_entropy",
        "timestamp": timestamp,
        "preset": preset_name,
        "layer": layer,
        "sae_id": sae_id,
        "ctx_len": int(args.ctx_len),
        "source_experiment_dir": str(source_dir),
        "n_features": int(n_feat),
        "n_pairs": int(len(pairs)),
        # Top-level scalars use |cosine|, the diagnostic predictor.
        "primary_predictor": "abs_cosine",
        "spearman_r": float(sp_r),
        "spearman_p": float(sp_p),
        "pearson_r": float(pe_r),
        "pearson_p": float(pe_p),
        "permutation_null_mean": float(null_mean),
        "permutation_null_std": float(null_std),
        "permutation_z": float(z),
        "permutation_p_two_sided": float(p_perm),
        "n_perm": int(n_perm),
        "stats_by_predictor": stats,
        "per_feature_neighbour": {
            "k": K,
            "spearman_r": float(nbr_sp.statistic),
            "spearman_p": float(nbr_sp.pvalue),
        },
        "neuronpedia_xcheck": {
            "n_overlap": int(len(xcheck)),
            "mean_abs_diff": float(xcheck["abs_diff"].mean()) if len(xcheck) else None,
            "max_abs_diff": float(xcheck["abs_diff"].max()) if len(xcheck) else None,
        },
    }
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2)
    print(f"[INFO] wrote manifest to {out_dir / 'manifest.json'}")

    # Symlink latest.
    latest = args.output_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(timestamp)
    print(f"[INFO] symlink: {latest} -> {timestamp}")
    print(f"[INFO] outputs in {out_dir}")


if __name__ == "__main__":
    main()
