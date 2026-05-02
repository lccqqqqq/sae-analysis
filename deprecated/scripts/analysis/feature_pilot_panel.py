"""Cherry-pick pilot driver: per-feature panel of entropies and metadata.

Reads a YAML of hand-picked features (see scripts/experiments/configs/) and
produces a single CSV row per feature containing:

  feature_id, tier, description,
  H_pos_mean, H_pos_std, H_pos_n_events,
  H_vocab, vocab_count,
  H_logit,
  density, neuronpedia_explanation, max_act_neuronpedia

H_pos is the event-averaged Shannon entropy of the per-position influence
distribution J_a(t') = ||grad f_a(t) / d x_{:,t'}||^2, computed only on
windows where the feature fires above ``threshold`` at the last position
(matching the convention in scripts/analysis/feature_token_influence.py).
The corpus is scanned until each feature has at least ``target_events`` or
``max_batches`` is reached.

H_vocab and H_logit come from feature_level_proxies; density and the
auto-interp explanation come from Neuronpedia (via scripts/utils).

Output: ``data/<preset>/feature_pilot_layer<L>.csv``
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import load_wikitext_train_text
from feature_level_proxies import (
    DEVICE, _resolve_unembed, compute_h_logit, compute_h_vocab,
)
from feature_token_influence import process_batch_with_influence
from model_adapters import load_model
from presets import get_preset
from sae_adapters import load_sae
from utils.neuronpedia_fetch import best_explanation, fetch_feature


def _entropy_bits(dist: np.ndarray) -> float:
    eps = 1e-12
    p = (dist + eps) / (dist.sum() + eps)
    return float(scipy.stats.entropy(p, base=2))


def collect_h_pos(
    feature_ids: list[int],
    model, sae, tokenizer, layer_idx, preset,
    threshold: float,
    context_len: int = 64,
    target_events: int = 20,
    max_batches: int = 4000,
    log_every: int = 200,
) -> dict[int, dict]:
    """Per-feature mean/std H_pos over firing events at window-last positions."""
    feature_set = set(feature_ids)
    per_feat: dict[int, list[float]] = {f: [] for f in feature_ids}

    text = load_wikitext_train_text()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total = tokens.shape[0]
    n_batches = min(total // context_len, max_batches)

    print(f"[INFO] H_pos: scanning up to {n_batches} windows of {context_len} "
          f"tokens (target {target_events} events/feature)", flush=True)
    t0 = time.time()
    for bi in range(n_batches):
        unfilled = {f for f in feature_set if len(per_feat[f]) < target_events}
        if not unfilled:
            break
        chunk = tokens[bi * context_len:(bi + 1) * context_len].to(DEVICE)
        try:
            infl = process_batch_with_influence(
                model, sae, chunk, layer_idx, unfilled, threshold, preset,
            )
        except Exception as e:  # individual window failure shouldn't kill the run
            print(f"[WARN] batch {bi}: {e}", flush=True)
            continue
        for fid, dist in infl.items():
            per_feat[fid].append(_entropy_bits(dist))

        if (bi + 1) % log_every == 0:
            rate = (bi + 1) / max(time.time() - t0, 1e-6)
            counts = [len(per_feat[f]) for f in feature_ids]
            print(f"  batch {bi+1}/{n_batches}  ({rate:.0f} b/s)"
                  f"  per-feat events min/median/max="
                  f"{min(counts)}/{int(np.median(counts))}/{max(counts)}",
                  flush=True)

    out = {}
    for fid in feature_ids:
        es = per_feat[fid]
        out[fid] = {
            "H_pos_mean": float(np.mean(es)) if es else float("nan"),
            "H_pos_std":  float(np.std(es))  if es else float("nan"),
            "H_pos_n_events": len(es),
        }
    print(f"[INFO] H_pos done in {time.time() - t0:.1f}s", flush=True)
    return out


def run_pilot(config_path: Path, output_csv: Path | None,
              context_len: int, target_events: int, max_batches: int,
              skip_neuronpedia: bool):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    preset = get_preset(cfg["preset"])
    layer = cfg["layer"]
    sae_id = cfg["sae_id"]
    feats = cfg["features"]
    feature_ids = [int(e["feature_id"]) for e in feats]
    tier_by_id = {int(e["feature_id"]): e["tier"] for e in feats}
    desc_by_id = {int(e["feature_id"]): e.get("description", "") for e in feats}

    print(f"[INFO] preset={preset.name} layer={layer} sae_id={sae_id}", flush=True)
    print(f"[INFO] {len(feature_ids)} features: {feature_ids}", flush=True)

    # Load once and share across all per-feature passes.
    model, tokenizer = load_model(preset, DEVICE)
    sae = load_sae(preset, layer, DEVICE)
    threshold = preset.threshold

    h_logit = compute_h_logit(feature_ids, sae, _resolve_unembed(model).to(DEVICE))

    h_vocab = compute_h_vocab(
        feature_ids, model, tokenizer, sae, layer, preset,
        threshold=threshold, context_len=context_len, max_batches=max_batches,
    )

    h_pos = collect_h_pos(
        feature_ids, model, sae, tokenizer, layer, preset,
        threshold=threshold, context_len=context_len,
        target_events=target_events, max_batches=max_batches,
    )

    # Neuronpedia metadata: density, auto-interp explanation. Optional so the
    # pilot can be re-run offline once cached.
    np_meta: dict[int, dict] = {}
    if not skip_neuronpedia:
        for fid in feature_ids:
            try:
                payload = fetch_feature(preset.name, sae_id, fid)
                np_meta[fid] = {
                    "density": payload.get("frac_nonzero"),
                    "explanation": best_explanation(payload),
                    "max_act": payload.get("maxActApprox"),
                }
            except Exception as e:
                print(f"[WARN] neuronpedia fetch for {fid} failed: {e}", flush=True)
                np_meta[fid] = {"density": None, "explanation": "", "max_act": None}
    else:
        np_meta = {fid: {"density": None, "explanation": "", "max_act": None}
                   for fid in feature_ids}

    if output_csv is None:
        out_dir = Path("data") / preset.name
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir / f"feature_pilot_layer{layer}.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "feature_id", "tier", "description",
            "H_pos_mean", "H_pos_std", "H_pos_n_events",
            "H_vocab", "vocab_count",
            "H_logit",
            "density", "neuronpedia_explanation", "max_act_neuronpedia",
        ])
        for fid in feature_ids:
            w.writerow([
                fid, tier_by_id[fid], desc_by_id[fid],
                h_pos[fid]["H_pos_mean"], h_pos[fid]["H_pos_std"],
                h_pos[fid]["H_pos_n_events"],
                h_vocab[fid]["H_vocab"], h_vocab[fid]["vocab_count"],
                h_logit[fid],
                np_meta[fid]["density"],
                np_meta[fid]["explanation"],
                np_meta[fid]["max_act"],
            ])
    print(f"[INFO] wrote {output_csv}")
    return output_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True,
                    help="YAML with preset/layer/sae_id/features.")
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--context-len", type=int, default=64)
    ap.add_argument("--target-events", type=int, default=20,
                    help="Per-feature minimum number of H_pos events.")
    ap.add_argument("--max-batches", type=int, default=4000,
                    help="Cap on corpus windows scanned.")
    ap.add_argument("--skip-neuronpedia", action="store_true")
    args = ap.parse_args()

    run_pilot(
        args.config, args.output_csv,
        context_len=args.context_len,
        target_events=args.target_events,
        max_batches=args.max_batches,
        skip_neuronpedia=args.skip_neuronpedia,
    )


if __name__ == "__main__":
    main()
