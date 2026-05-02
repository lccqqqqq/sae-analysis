"""Compute H_pos on the top-activating contexts of cherry-picked features.

For each (feature_id, top-activation-rank) saved in the JSON produced by
``scripts/analysis/top_activating_contexts.py``, this script:

  1. Recovers the absolute corpus token index of the activating-token peak
     (``window_start_token + token_idx`` from the JSON).
  2. Slices ``tokens[abs_peak - ctx_len + 1 : abs_peak + 1]`` from the
     same wikitext corpus used by the rest of the pipeline, so the peak
     token is at the *last* position of the window — matching the
     gradient-pipeline convention in scripts/analysis/feature_token_influence.py.
  3. Runs ``process_batch_with_influence`` to obtain the per-position
     gradient-norm-squared distribution J_a(t').
  4. Computes H_pos = entropy(J / sum(J), base 2).

A ``--tier-filter`` flag lets the script process only one tier of features
from the YAML at a time, so multiple instances can run in parallel on
separate GPUs and each writes its own CSV without contention.

Outputs (under ``--output-dir``):
  - ctx<N>_<tier>.csv         per-event scalars (see HEADER below)
  - influences_<tier>.npz     raw J_a(t') vectors keyed by
                              "<feature_id>_<top_act_rank>"; reusable for
                              downstream Renyi / participation-ratio /
                              percentile analyses without re-running.

Used by scripts/experiments/submit_cherry_picked_entropy.sh.
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

from data_loader import load_wikitext_train_text
from feature_token_influence import process_batch_with_influence
from model_adapters import load_model
from presets import get_preset
from sae_adapters import load_sae

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

HEADER = [
    "feature_id", "tier", "description",
    "top_act_rank", "activation_value",
    "abs_peak", "ctx_len", "ctx_start",
    "H_pos_bits",
    "n_positions", "J_max", "J_sum", "J_argmax",
    "peak_token_str", "status",
]


def _entropy_bits(dist: np.ndarray) -> float:
    eps = 1e-12
    p = (dist + eps) / (dist.sum() + eps)
    return float(scipy.stats.entropy(p, base=2))


def run(
    yaml_path: Path,
    top_contexts_path: Path,
    output_dir: Path,
    ctx_lens: list[int],
    top_k: int,
    tier_filter: str | None,
    threshold: float | None,
    corpus_tensor_path: Path | None = None,
):
    cfg = yaml.safe_load(yaml_path.read_text())
    preset = get_preset(cfg["preset"])
    layer_idx = cfg["layer"]
    threshold = threshold if threshold is not None else preset.threshold

    feats = cfg["features"]
    if tier_filter:
        feats = [e for e in feats if e["tier"] == tier_filter]
    if not feats:
        print(f"[INFO] no features after tier_filter={tier_filter!r}; nothing to do")
        return

    feature_ids = [int(e["feature_id"]) for e in feats]
    tier_by_id = {int(e["feature_id"]): e["tier"] for e in feats}
    desc_by_id = {int(e["feature_id"]): e.get("description", "") for e in feats}

    # Top-contexts JSON keyed by str(feature_id).
    import json
    top_contexts = json.loads(top_contexts_path.read_text())

    # Load model + SAE + corpus tokenization once.
    model, tokenizer = load_model(preset, DEVICE)
    sae = load_sae(preset, layer_idx, DEVICE)
    print(f"[INFO] preset={preset.name} layer={layer_idx} arch={sae.arch} "
          f"n_latent={sae.n_latent}", flush=True)

    if corpus_tensor_path is not None:
        corpus_tokens = torch.load(corpus_tensor_path).to(torch.long)
        if corpus_tokens.dim() != 1:
            raise ValueError(f"corpus_tensor must be 1-D, got shape "
                             f"{tuple(corpus_tokens.shape)}")
        print(f"[INFO] corpus_tokens loaded from {corpus_tensor_path}",
              flush=True)
    else:
        text = load_wikitext_train_text()
        corpus_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    n_corpus = corpus_tokens.shape[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] corpus tokens: {n_corpus:,}  ctx_lens={ctx_lens}  "
          f"top_k={top_k}  tier_filter={tier_filter}", flush=True)

    for ctx_len in ctx_lens:
        _run_one_ctx_len(
            ctx_len=ctx_len, feature_ids=feature_ids,
            tier_by_id=tier_by_id, desc_by_id=desc_by_id,
            top_contexts=top_contexts, corpus_tokens=corpus_tokens,
            n_corpus=n_corpus, top_k=top_k, tier_filter=tier_filter,
            output_dir=output_dir, model=model, sae=sae,
            layer_idx=layer_idx, preset=preset, threshold=threshold,
        )


def _run_one_ctx_len(
    *, ctx_len, feature_ids, tier_by_id, desc_by_id, top_contexts,
    corpus_tokens, n_corpus, top_k, tier_filter, output_dir,
    model, sae, layer_idx, preset, threshold,
):
    print(f"\n[INFO] === ctx_len={ctx_len} ===", flush=True)
    csv_name = f"ctx{ctx_len:03d}_{tier_filter or 'all'}.csv"
    csv_path = output_dir / csv_name
    npz_name = f"influences_ctx{ctx_len:03d}_{tier_filter or 'all'}.npz"
    npz_path = output_dir / npz_name

    rows: list[list] = []
    influences: dict[str, np.ndarray] = {}

    t0 = time.time()
    for fid in feature_ids:
        per_feat_examples = (top_contexts.get(str(fid)) or {}).get("examples", [])
        if not per_feat_examples:
            print(f"[WARN] feature {fid}: no top-activations in JSON", flush=True)
            continue
        per_feat_examples = per_feat_examples[:top_k]

        for rank, ex in enumerate(per_feat_examples):
            abs_peak = int(ex["window_start_token"]) + int(ex["token_idx"])
            ctx_start = abs_peak - ctx_len + 1
            base_row = {
                "feature_id": fid,
                "tier": tier_by_id[fid],
                "description": desc_by_id[fid],
                "top_act_rank": rank,
                "activation_value": float(ex["activation"]),
                "abs_peak": abs_peak,
                "ctx_len": ctx_len,
                "ctx_start": ctx_start,
                "H_pos_bits": float("nan"),
                "n_positions": 0,
                "J_max": float("nan"),
                "J_sum": float("nan"),
                "J_argmax": -1,
                "peak_token_str": ex.get("activating_token", ""),
                "status": "",
            }
            if ctx_start < 0 or abs_peak >= n_corpus:
                base_row["status"] = "peak_too_close_to_corpus_boundary"
                rows.append([base_row[k] for k in HEADER])
                continue

            window = corpus_tokens[ctx_start:abs_peak + 1].to(DEVICE)
            try:
                infl = process_batch_with_influence(
                    model, sae, window, layer_idx,
                    leading_features={fid},
                    threshold=threshold,
                    preset=preset,
                )
            except Exception as e:
                base_row["status"] = f"compute_error: {e!r}"[:80]
                rows.append([base_row[k] for k in HEADER])
                continue

            if fid not in infl:
                base_row["status"] = "feature_below_threshold_at_peak"
                rows.append([base_row[k] for k in HEADER])
                continue

            J = infl[fid]
            base_row["H_pos_bits"] = _entropy_bits(J)
            base_row["n_positions"] = int(J.shape[0])
            base_row["J_max"] = float(J.max())
            base_row["J_sum"] = float(J.sum())
            base_row["J_argmax"] = int(J.argmax())
            base_row["status"] = "ok"
            rows.append([base_row[k] for k in HEADER])
            influences[f"{fid}_{rank}"] = J.astype(np.float32)

        n_done = sum(1 for r in rows if r[0] == fid)
        n_ok = sum(1 for r in rows if r[0] == fid and r[HEADER.index("status")] == "ok")
        print(f"[INFO] feature {fid} ({tier_by_id[fid]}): "
              f"{n_ok}/{n_done} events ok", flush=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    np.savez_compressed(npz_path, **influences)

    elapsed = time.time() - t0
    n_ok = sum(1 for r in rows if r[HEADER.index("status")] == "ok")
    print(f"[INFO] wrote {csv_path}  ({len(rows)} rows, {n_ok} ok)", flush=True)
    print(f"[INFO] wrote {npz_path}  ({len(influences)} influence vectors)",
          flush=True)
    print(f"[INFO] elapsed: {elapsed:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", type=Path, required=True,
                    help="Pilot YAML with feature_id/tier/description.")
    ap.add_argument("--top-contexts", type=Path, required=True,
                    help="JSON from top_activating_contexts.py.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Per-experiment timestamped dir; CSV + NPZ go here.")
    ap.add_argument("--ctx-len", type=int, nargs="+", default=[64],
                    help="One or more context lengths (e.g. --ctx-len 64 "
                         "or --ctx-len 16 32 64 96 128 for a sweep). "
                         "Model and corpus are loaded once and reused.")
    ap.add_argument("--top-k", type=int, default=8,
                    help="Use the top K activations per feature.")
    ap.add_argument("--tier-filter", type=str, default=None,
                    choices=["token", "phrase", "concept", "abstract"],
                    help="If set, process only features with this tier.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Activation threshold for the gradient gate "
                         "(default: preset.threshold).")
    ap.add_argument("--corpus-tensor", type=Path, default=None,
                    help="If set, load this 1-D LongTensor as the corpus "
                         "instead of re-tokenising wikitext. Used by the "
                         "Neuronpedia adapter.")
    args = ap.parse_args()
    run(
        yaml_path=args.yaml,
        top_contexts_path=args.top_contexts,
        output_dir=args.output_dir,
        ctx_lens=list(args.ctx_len),
        top_k=args.top_k,
        tier_filter=args.tier_filter,
        threshold=args.threshold,
        corpus_tensor_path=args.corpus_tensor,
    )


if __name__ == "__main__":
    main()
