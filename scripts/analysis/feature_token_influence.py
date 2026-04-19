"""
Per-feature gradient-based influence on the token sequence.

Refactored to dispatch model wiring and SAE loading through adapters, so the
same code path works for any (model, SAE) pair declared in presets.PRESETS.

Core entry point used downstream:
    process_batch_with_influence(model, sae, tokens, layer_idx,
                                 leading_features, threshold, preset)
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from model_adapters import DummyEmbed, get_embed, get_layer, load_model, set_embed
from presets import Preset, get_preset, site_for
from sae_adapters import SAEBundle, load_sae

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
THRESHOLD = 1.0
BATCH_SIZE = 64
MAX_BATCHES = 5000
MIN_FEATURE_ACTIVATIONS = 10
CHECKPOINT_INTERVAL = 100


# --- Core Influence Computation Functions ---

def compute_influence_for_feature(feature_activation, input_embeds):
    """Compute J(t' | t) = ||∂f_a(t) / ∂x_{:, t'}||^2 via a single backward pass.

    Returns a [seq_len] tensor of per-position squared-L2 norms.
    """
    if input_embeds.grad is not None:
        input_embeds.grad.zero_()

    feature_activation.backward(retain_graph=True)

    grads = input_embeds.grad
    if grads is None:
        return torch.zeros(input_embeds.shape[1], device=input_embeds.device)
    grads = grads.clone()
    influence_norms = (grads ** 2).sum(dim=-1)  # [1, seq_len]
    return influence_norms[0]


def process_batch_with_influence(
    model, sae: SAEBundle, tokens, layer_idx, leading_features, threshold,
    preset: Preset,
):
    """Compute per-feature influence distributions on a single (batch, layer).

    Args:
        model: HF causal LM (on DEVICE).
        sae:   SAEBundle from sae_adapters.load_sae.
        tokens: token IDs, shape [seq_len].
        layer_idx: which transformer layer to hook.
        leading_features: iterable of feature indices to (try to) compute
            influence for. Only those active at the last position > threshold
            get a backward pass.
        threshold: activation cutoff for "active".
        preset: Preset specifying model-attribute paths.

    Returns:
        dict feat_idx -> influence distribution (np.ndarray, [seq_len]).
    """
    input_ids = tokens.unsqueeze(0)  # [1, seq_len]

    # Embed the inputs through the real embedding layer, detach, make leaf.
    embed_layer = get_embed(model, preset)
    input_embeds = embed_layer(input_ids)
    if isinstance(input_embeds, tuple):
        input_embeds = input_embeds[0]
    input_embeds = input_embeds.detach()
    input_embeds.requires_grad_(True)

    # Hook to capture residual stream at layer_idx.
    layer = get_layer(model, preset, layer_idx)
    activations = []

    def hook_fn(module, inputs, output):  # noqa: ARG001
        activations.append(output[0] if isinstance(output, tuple) else output)

    handle = layer.register_forward_hook(hook_fn)

    # Swap in DummyEmbed so the forward pass routes our leaf tensor.
    original_embed = get_embed(model, preset)
    set_embed(model, preset, DummyEmbed(input_embeds))

    try:
        _ = model(input_ids)

        resid = activations[0]            # [1, seq_len, d_model]
        feats = sae.encode(resid)         # [1, seq_len, n_latent]
        last_pos_feats = feats[0, -1, :]  # [n_latent]

        feature_influences = {}
        for feat_idx in leading_features:
            if last_pos_feats[feat_idx] > threshold:
                feat_activation = last_pos_feats[feat_idx]
                influence_norms = compute_influence_for_feature(
                    feat_activation, input_embeds
                )
                feature_influences[feat_idx] = influence_norms.detach().cpu().numpy()
    finally:
        set_embed(model, preset, original_embed)
        handle.remove()

    return feature_influences


# --- Checkpoint I/O ---------------------------------------------------------

def load_checkpoint(checkpoint_file):
    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        return None
    print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
    data = torch.load(checkpoint_path, map_location="cpu")
    influence_data = defaultdict(list)
    for feat_idx, influence_list in data.get("influence_data", {}).items():
        influence_data[feat_idx] = influence_list
    return {
        "influence_data": influence_data,
        "batch_count": data.get("batch_count", 0),
        "start_idx": data.get("start_idx", 0),
        "config": data.get("config", {}),
    }


def save_checkpoint(checkpoint_file, influence_data, batch_count, start_idx, config):
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "influence_data": dict(influence_data),
        "batch_count": batch_count,
        "start_idx": start_idx,
        "config": config,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path} ({batch_count} batches)")


# --- Main Analysis (standalone entry point) ---------------------------------

def main(preset_name="pythia-70m", layer_idx=3, sparsity_file=None,
         output_file=None, checkpoint_file=None, resume=True,
         threshold=None, max_batches=None):
    preset = get_preset(preset_name)
    site = site_for(preset, layer_idx)
    threshold = threshold if threshold is not None else THRESHOLD
    max_batches = max_batches if max_batches is not None else MAX_BATCHES

    print(f"\n{'='*60}")
    print(f"[INFO] Processing preset={preset.name} site={site}")
    print(f"{'='*60}")

    model, tokenizer = load_model(preset, DEVICE)
    sae = load_sae(preset, layer_idx, DEVICE)
    print(f"[INFO] SAE: arch={sae.arch} n_latent={sae.n_latent} source={sae.source}")

    # Leading features: top-frequency set from feature_sparsity_data_<site>.pt.
    if sparsity_file is None:
        sparsity_file = Path(f"feature_sparsity_data_{site}.pt")
        if not sparsity_file.exists():
            sparsity_file = Path("feature_sparsity_data.pt")

    if not Path(sparsity_file).exists():
        print(f"[ERROR] {sparsity_file} not found. Run feature_sparsity.py first.")
        sys.exit(1)

    sparsity_data = torch.load(sparsity_file)
    frequencies = sparsity_data["frequencies"]

    MIN_FREQ = 0.001
    MAX_FEATURES = 500
    leading_mask = frequencies > MIN_FREQ
    candidate_indices = torch.nonzero(leading_mask).squeeze(-1)
    if len(candidate_indices) > MAX_FEATURES:
        candidate_freqs = frequencies[candidate_indices]
        _, top_indices = torch.topk(candidate_freqs, MAX_FEATURES, largest=True)
        leading_features = set(candidate_indices[top_indices].tolist())
        print(f"[INFO] {len(candidate_indices)} candidates; keeping top {MAX_FEATURES}")
    else:
        leading_features = set(candidate_indices.tolist())
    print(f"[INFO] Selected {len(leading_features)} leading features (freq > {MIN_FREQ:.1%})")

    from data_loader import load_wikitext_train_text
    text = load_wikitext_train_text()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_tokens = tokens.shape[0]
    print(f"[INFO] Total tokens: {total_tokens}")

    if checkpoint_file is None:
        checkpoint_file = Path(f"feature_token_influence_{site}_checkpoint.pt")
    else:
        checkpoint_file = Path(checkpoint_file)

    influence_data = defaultdict(list)
    start_idx = 0
    batch_count = 0
    if resume and checkpoint_file.exists():
        ck = load_checkpoint(checkpoint_file)
        if ck:
            influence_data = ck["influence_data"]
            batch_count = ck["batch_count"]
            start_idx = ck["start_idx"]
            print(f"[INFO] Resuming: batch={batch_count} start={start_idx}")

    print(f"[INFO] Computing influence (threshold > {threshold})...")
    t0 = time.time()
    for i in range(start_idx, total_tokens - BATCH_SIZE, BATCH_SIZE):
        if batch_count >= max_batches:
            break
        chunk = tokens[i: i + BATCH_SIZE].to(DEVICE)
        try:
            batch_influences = process_batch_with_influence(
                model, sae, chunk, layer_idx, leading_features, threshold, preset
            )
            for feat_idx, dist in batch_influences.items():
                influence_data[feat_idx].append(dist)
        except Exception as e:
            print(f"[WARN] batch {batch_count}: {e}")
            continue

        batch_count += 1
        if batch_count % 10 == 0:
            print(f"Processed {batch_count}/{max_batches} batches...", end="\r")
        if batch_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                checkpoint_file, influence_data, batch_count, i + BATCH_SIZE,
                {"threshold": threshold, "layer": layer_idx, "site": site,
                 "batch_size": BATCH_SIZE, "min_freq": MIN_FREQ,
                 "preset": preset.name},
            )

    print(f"\nProcessed {batch_count} batches in {time.time()-t0:.1f}s.")

    # Aggregate
    aggregated = {}
    for feat_idx, influence_list in influence_data.items():
        if len(influence_list) >= MIN_FEATURE_ACTIVATIONS:
            arr = np.stack(influence_list, axis=0)
            aggregated[feat_idx] = {
                "mean_influence": arr.mean(axis=0).tolist(),
                "std_influence": arr.std(axis=0).tolist(),
                "all_influences": arr.tolist(),
                "num_samples": len(influence_list),
            }

    output_data = {
        "feature_influences": aggregated,
        "config": {
            "preset": preset.name, "threshold": threshold, "layer": layer_idx,
            "site": site, "total_batches": batch_count, "batch_size": BATCH_SIZE,
            "min_freq": MIN_FREQ,
        },
    }
    if output_file is None:
        output_file = Path(f"feature_token_influence_{site}.pt")
    torch.save(output_data, Path(output_file))
    print(f"[INFO] Saved results to {output_file}")

    if checkpoint_file.exists():
        checkpoint_file.unlink()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Per-feature token influence")
    parser.add_argument("--preset", type=str, default="pythia-70m")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--sparsity-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    main(
        preset_name=args.preset, layer_idx=args.layer,
        sparsity_file=args.sparsity_file, output_file=args.output_file,
        checkpoint_file=args.checkpoint_file, resume=not args.no_resume,
        threshold=args.threshold, max_batches=args.max_batches,
    )
