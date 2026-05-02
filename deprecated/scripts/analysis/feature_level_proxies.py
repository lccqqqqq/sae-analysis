"""Per-feature unsupervised proxies for "abstraction level".

Two scalars are computed for each feature in a user-supplied subset:

    H_vocab(a) = H[ p_a(v) ],  p_a(v) = c_a(v) / T_a
        where c_a(v) is the count of corpus tokens v at which feature a
        fires above ``threshold``, and T_a = sum_v c_a(v).
        Low  -> feature triggers on a narrow vocabulary slice (token-like).
        High -> feature triggers on a broad vocabulary distribution
        (concept-like or topical).

    H_logit(a) = H[ softmax(W_U D_a) ]
        where D_a is feature a's decoder column (shape [d_model]) and W_U is
        the model's output-embedding matrix.
        Low  -> feature pushes a sharp / token-specific output distribution.
        High -> feature pushes a broad output distribution.

Both are single scalars per feature, no batch averaging on the H_logit side
and a single corpus sweep on the H_vocab side. The implementation is
restricted to a subset of feature indices to keep the pilot cheap; the full
16k-feature variant is a straightforward generalisation.

Output (when run standalone): a CSV with columns
    feature_id, H_vocab, H_logit, vocab_count

Used by feature_pilot_panel.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_wikitext_train_text
from model_adapters import get_layer, load_model
from presets import Preset, get_preset
from sae_adapters import SAEBundle, load_sae

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def _resolve_unembed(model: torch.nn.Module) -> torch.Tensor:
    """Return the model's output-embedding matrix as [vocab, d_model] float32.

    Tries ``lm_head.weight`` and ``embed_out.weight`` first; falls back to
    the tied input embedding via ``get_input_embeddings``.
    """
    for path in ("lm_head.weight", "embed_out.weight"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.Tensor) and obj.dim() == 2:
            return obj.detach().float()
    return model.get_input_embeddings().weight.detach().float()


def compute_h_logit(
    feature_ids: list[int], sae: SAEBundle, unembed_w: torch.Tensor,
) -> dict[int, float]:
    """H_logit(a) for each requested feature, in bits."""
    inv_ln2 = 1.0 / math.log(2.0)
    decoder_cols = sae.dec_w[:, feature_ids]              # [d_model, k]
    logits = unembed_w @ decoder_cols.to(unembed_w.device)  # [vocab, k]
    log_probs = F.log_softmax(logits, dim=0)
    probs = log_probs.exp()
    H_nats = -(probs * log_probs).sum(dim=0)              # [k]
    H_bits = (H_nats * inv_ln2).cpu().tolist()
    return dict(zip(feature_ids, H_bits))


def _entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy in bits from an integer count vector. NaN on T==0."""
    T = counts.sum()
    if T <= 0:
        return float("nan")
    nz = counts[counts > 0].astype(np.float64)
    p = nz / T
    return float(-(p * np.log2(p)).sum())


def compute_h_vocab(
    feature_ids: list[int],
    model: torch.nn.Module,
    tokenizer,
    sae: SAEBundle,
    layer_idx: int,
    preset: Preset,
    threshold: float,
    context_len: int = 128,
    max_batches: int | None = None,
    log_every: int = 200,
) -> dict[int, dict]:
    """Stream the corpus, count token-level firings per requested feature.

    Returns ``{feature_id: {"H_vocab": float, "vocab_count": int}}``. Only
    the requested feature_ids contribute to the count tensor, so memory is
    O(k * vocab) regardless of SAE width.
    """
    device = sae.enc_w.device
    feature_ids_t = torch.tensor(feature_ids, dtype=torch.long, device=device)
    feat_pos = {fid: i for i, fid in enumerate(feature_ids)}
    k = len(feature_ids)
    # Use the model's actual embedding size, not tokenizer.vocab_size — for
    # several modern tokenizers (Gemma-2, Llama-3) the two disagree by a few
    # hundred padding slots and the larger one is the safe upper bound.
    vocab_size = model.get_input_embeddings().num_embeddings

    counts = torch.zeros(k, vocab_size, dtype=torch.int64, device=device)

    # Capture the residual stream at layer_idx via a forward hook.
    captured: list[torch.Tensor] = []

    def hook_fn(module, inputs, output):  # noqa: ARG001
        captured.append(output[0] if isinstance(output, tuple) else output)

    layer = get_layer(model, preset, layer_idx)
    handle = layer.register_forward_hook(hook_fn)

    text = load_wikitext_train_text()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total = tokens.shape[0]
    n_batches = total // context_len
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    print(f"[INFO] H_vocab: streaming {n_batches} windows of {context_len} tokens "
          f"for {k} features, vocab={vocab_size}", flush=True)
    t0 = time.time()
    try:
        for bi in range(n_batches):
            chunk = tokens[bi * context_len:(bi + 1) * context_len].to(device)
            captured.clear()
            with torch.no_grad():
                _ = model(chunk.unsqueeze(0))
            resid = captured[0]                            # [1, T, d_model]
            feats = sae.encode(resid)[0, :, feature_ids_t]  # [T, k]
            active = feats > threshold                     # [T, k]
            pos_idx, k_idx = active.nonzero(as_tuple=True)
            if pos_idx.numel() == 0:
                continue
            tok_ids = chunk[pos_idx]
            flat = counts.view(-1)
            flat_idx = k_idx.to(torch.int64) * vocab_size + tok_ids.to(torch.int64)
            flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))

            if (bi + 1) % log_every == 0:
                rate = (bi + 1) / max(time.time() - t0, 1e-6)
                eta = (n_batches - bi - 1) / rate
                print(f"  batch {bi+1}/{n_batches}  ({rate:.0f} b/s, eta {eta:.0f}s)",
                      flush=True)
    finally:
        handle.remove()

    counts_np = counts.cpu().numpy()
    out: dict[int, dict] = {}
    for fid in feature_ids:
        i = feat_pos[fid]
        H = _entropy_from_counts(counts_np[i])
        out[fid] = {"H_vocab": H, "vocab_count": int(counts_np[i].sum())}
    print(f"[INFO] H_vocab done in {time.time() - t0:.1f}s", flush=True)
    return out


def compute_level_proxies(
    feature_ids: list[int],
    preset: Preset,
    layer_idx: int,
    threshold: float | None = None,
    context_len: int = 128,
    max_batches: int | None = None,
) -> dict[int, dict]:
    """High-level entry: returns {feature_id: {H_vocab, H_logit, vocab_count}}."""
    if threshold is None:
        threshold = preset.threshold

    model, tokenizer = load_model(preset, DEVICE)
    sae = load_sae(preset, layer_idx, DEVICE)
    unembed_w = _resolve_unembed(model).to(DEVICE)

    h_logit = compute_h_logit(feature_ids, sae, unembed_w)
    h_vocab = compute_h_vocab(
        feature_ids, model, tokenizer, sae, layer_idx, preset,
        threshold=threshold, context_len=context_len, max_batches=max_batches,
    )

    return {
        fid: {
            "H_vocab": h_vocab[fid]["H_vocab"],
            "vocab_count": h_vocab[fid]["vocab_count"],
            "H_logit": h_logit[fid],
        }
        for fid in feature_ids
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="gemma-2-2b")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--feature-ids", type=int, nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--output-csv", type=Path, default=None)
    args = ap.parse_args()

    preset = get_preset(args.preset)
    out = compute_level_proxies(
        args.feature_ids, preset, args.layer,
        threshold=args.threshold,
        context_len=args.context_len,
        max_batches=args.max_batches,
    )

    if args.output_csv is None:
        out_dir = Path("data") / preset.name
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_csv = out_dir / f"level_proxies_layer{args.layer}.csv"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_id", "H_vocab", "H_logit", "vocab_count"])
        for fid in args.feature_ids:
            r = out[fid]
            w.writerow([fid, r["H_vocab"], r["H_logit"], r["vocab_count"]])
    print(f"[INFO] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
