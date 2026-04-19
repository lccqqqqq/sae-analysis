"""Compute per-feature "level" entropies (H_vocab and H_logit) for a preset.

Two independent feature-intrinsic quantities are produced for every SAE
feature at every layer of the chosen preset:

    H_vocab(a) = entropy of the distribution over input tokens v at which
                 feature a fires above threshold. Low = token-specific;
                 high = topical / abstract.

    H_logit(a) = entropy of softmax(W_U @ D_a), where D_a is the feature's
                 decoder column and W_U the model's output embedding.
                 Low = feature pushes a narrow output distribution;
                 high = broad / semantic.

Both are single scalars per feature -- no batch-averaging, no gradients.
Together with the batch-averaged influence entropy from the existing
entropy_comparison_*.pt files, this lets us test whether feature "level"
(as proxied by H_vocab or H_logit) correlates with nonlocality (H_feat).

Output: data/<preset>/feature_level_entropies.pt
    {
      "layer_<N>": {
        "H_vocab":  np.ndarray [n_latent]   (nan where feature has < min_count fires)
        "counts":   np.ndarray [n_latent]   (total count used for H_vocab)
        "H_logit":  np.ndarray [n_latent]
      },
      ...
      "config": {...},
    }

Usage (on a GPU node):
    python scripts/analysis/feature_level_entropies.py \
        --preset pythia-70m --layers 0 1 2 3 4 5 --threshold 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_adapters import get_layer, load_model  # noqa: E402
from presets import get_preset  # noqa: E402
from sae_adapters import load_sae  # noqa: E402

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def compute_logit_entropy(decoder_w: torch.Tensor,
                          unembed_w: torch.Tensor,
                          chunk: int = 512) -> torch.Tensor:
    """H_logit(a) for each feature a via softmax(W_U @ D_a).

    decoder_w: [d_model, n_latent]   (note: sae_adapters stores dec_w this way)
    unembed_w: [vocab_size, d_model]
    returns : [n_latent]   (float32, bits)
    """
    n_latent = decoder_w.shape[1]
    device = decoder_w.device
    out = torch.zeros(n_latent, dtype=torch.float32, device=device)
    inv_ln2 = 1.0 / math.log(2.0)
    for i in range(0, n_latent, chunk):
        block = decoder_w[:, i:i + chunk]                       # [d_model, c]
        logits = unembed_w @ block                              # [vocab_size, c]
        log_probs = F.log_softmax(logits, dim=0)
        probs = log_probs.exp()
        H_nats = -(probs * log_probs).sum(dim=0)                # [c]
        out[i:i + chunk] = H_nats * inv_ln2
    return out


def compute_vocab_counts(
    model, sae, tokens: torch.Tensor, layer_idx: int, preset,
    threshold: float, batch_size: int, vocab_size: int,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Run SAE over the corpus and accumulate per-feature token-count matrix.

    Returns counts [n_latent, vocab_size] as int32 on the SAE device.
    """
    n_latent = sae.n_latent
    device = sae.enc_w.device
    counts = torch.zeros(n_latent, vocab_size, dtype=torch.int32, device=device)

    layer = get_layer(model, preset, layer_idx)
    captured: list = []

    def hook_fn(module, inputs, output):  # noqa: ARG001
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = layer.register_forward_hook(hook_fn)

    total_tokens = tokens.shape[0]
    n_batches = total_tokens // batch_size
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    t0 = time.time()
    try:
        for bi in range(n_batches):
            chunk = tokens[bi * batch_size:(bi + 1) * batch_size].to(device)
            captured.clear()
            with torch.no_grad():
                _ = model(chunk.unsqueeze(0))
            resid = captured[0]                          # [1, seq, d_model]
            feats = sae.encode(resid)                    # [1, seq, n_latent]
            active = feats[0] > threshold                # [seq, n_latent]
            pos_idx, feat_idx = active.nonzero(as_tuple=True)
            if pos_idx.numel() == 0:
                continue
            tok_ids = chunk[pos_idx]
            flat = counts.view(-1)
            flat_idx = (feat_idx.to(torch.int64) * vocab_size
                        + tok_ids.to(torch.int64))
            flat.scatter_add_(0, flat_idx,
                              torch.ones_like(flat_idx, dtype=torch.int32))

            if (bi + 1) % 1000 == 0:
                rate = (bi + 1) / max(time.time() - t0, 1e-6)
                eta = (n_batches - bi - 1) / rate
                print(f"    layer {layer_idx}: batch {bi+1}/{n_batches}"
                      f" ({rate:.0f} b/s, eta {eta:.0f}s)", flush=True)
    finally:
        handle.remove()

    print(f"    layer {layer_idx}: done in {time.time()-t0:.1f}s", flush=True)
    return counts


def vocab_entropy_from_counts(counts: torch.Tensor,
                              min_count: int = 20,
                              chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """H_vocab per feature from [n_latent, vocab_size] counts.

    Using the closed form  H = log2(T) - (1/T) * sum_v c_v * log2(c_v).
    Computed in feature-chunks to avoid blowing up GPU memory (the float32
    expansion would be 4x the int32 counts size).
    Returns (H [n_latent], totals [n_latent]) on CPU; NaN where T < min_count.
    """
    n_latent = counts.shape[0]
    inv_ln2 = 1.0 / math.log(2.0)
    H_np = np.zeros(n_latent, dtype=np.float32)
    T_np = np.zeros(n_latent, dtype=np.int64)
    for i in range(0, n_latent, chunk):
        block = counts[i:i + chunk].to(torch.float32)              # [c, vocab]
        totals = block.sum(dim=-1)                                  # [c]
        safe_T = totals.clamp(min=1.0)
        # c log c is zero where c==0; clamp avoids log(0) NaN
        c_log_c = (block * torch.log(block.clamp(min=1.0)) * inv_ln2).sum(dim=-1)
        H = torch.log2(safe_T) - c_log_c / safe_T
        H_np[i:i + chunk] = H.cpu().numpy().astype(np.float32)
        T_np[i:i + chunk] = totals.cpu().numpy().astype(np.int64)
        del block, totals, safe_T, c_log_c, H
    H_np[T_np < min_count] = np.nan
    return H_np, T_np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", type=str, default="pythia-70m")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Subset of layers; default = all for the preset.")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="Activation threshold for H_vocab counting.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--min-count", type=int, default=20,
                    help="Min total activations to trust an H_vocab estimate.")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Cap on corpus batches (debug only).")
    ap.add_argument("--output-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    preset = get_preset(args.preset)
    layers = args.layers or list(range(preset.num_layers))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[INFO] preset={preset.name}  device={DEVICE}  layers={layers}", flush=True)
    print(f"[INFO] threshold={args.threshold}  batch_size={args.batch_size}", flush=True)

    model, tokenizer = load_model(preset, DEVICE)
    vocab_size = tokenizer.vocab_size

    # Unembedding weight (tied across many archs; for Pythia, model.embed_out.weight).
    unembed_w = None
    for attr_chain in ("embed_out.weight", "lm_head.weight",
                       "transformer.ln_f.weight",  # fallback - not ideal
                       ):
        obj = model
        try:
            for part in attr_chain.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.Tensor) and obj.dim() == 2:
                unembed_w = obj
                print(f"[INFO] unembed from model.{attr_chain}  shape={tuple(obj.shape)}",
                      flush=True)
                break
        except AttributeError:
            continue
    if unembed_w is None:
        # Fall back to tied embeddings
        embed = model.get_input_embeddings().weight
        unembed_w = embed
        print(f"[INFO] using tied input embedding  shape={tuple(embed.shape)}",
              flush=True)
    unembed_w = unembed_w.detach().to(DEVICE).float()
    vocab_size = unembed_w.shape[0]

    # Load + tokenize corpus
    from data_loader import load_wikitext_train_text
    tokens = tokenizer(load_wikitext_train_text(),
                       return_tensors="pt")["input_ids"][0]
    print(f"[INFO] corpus tokens: {tokens.shape[0]:,}", flush=True)

    out_dir = args.output_dir / preset.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_level_entropies.pt"

    result = {
        "config": {
            "preset": preset.name, "model_id": preset.model_id,
            "layers": layers, "threshold": args.threshold,
            "batch_size": args.batch_size, "min_count": args.min_count,
            "timestamp": timestamp, "vocab_size": vocab_size,
            "corpus_tokens": int(tokens.shape[0]),
        },
    }

    for L in layers:
        print(f"\n[INFO] --- layer {L} ---", flush=True)
        sae = load_sae(preset, L, DEVICE)
        n_latent = sae.n_latent
        print(f"    arch={sae.arch}  n_latent={n_latent}", flush=True)

        # H_logit: cheap, no corpus pass
        t0 = time.time()
        H_logit = compute_logit_entropy(sae.dec_w, unembed_w).cpu().numpy()
        print(f"    H_logit done ({time.time()-t0:.1f}s)"
              f"  mean={np.nanmean(H_logit):.3f}  std={np.nanstd(H_logit):.3f}",
              flush=True)

        # H_vocab: corpus pass
        counts = compute_vocab_counts(
            model, sae, tokens, L, preset,
            threshold=args.threshold, batch_size=args.batch_size,
            vocab_size=vocab_size, max_batches=args.max_batches,
        )
        H_vocab, totals = vocab_entropy_from_counts(counts, min_count=args.min_count)
        n_valid = int(np.sum(~np.isnan(H_vocab)))
        print(f"    H_vocab: {n_valid}/{n_latent} features above min_count"
              f"  mean={np.nanmean(H_vocab):.3f}  std={np.nanstd(H_vocab):.3f}",
              flush=True)

        result[f"layer_{L}"] = {
            "H_logit": H_logit,
            "H_vocab": H_vocab,
            "totals":  totals,
        }

        # Save incrementally so the job can be killed without losing work
        torch.save(result, out_path)
        print(f"    saved -> {out_path}", flush=True)

        # Free SAE tensors before next iteration
        del sae, counts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final meta
    with open(out_dir / "feature_level_entropies_config.json", "w") as f:
        json.dump(result["config"], f, indent=2)
    print(f"\n[DONE] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
