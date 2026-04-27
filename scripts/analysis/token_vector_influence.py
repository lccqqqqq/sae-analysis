"""
Influence of the residual-stream (token) vector on upstream token positions,
used as a baseline against per-feature influence.

Refactored to dispatch through adapters; no model or site strings are hard-coded.
"""

import time
from pathlib import Path

import numpy as np
import scipy.stats
import torch

from model_adapters import DummyEmbed, get_embed, get_layer, load_model, set_embed
from presets import Preset, get_preset, site_for

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
BATCH_SIZE = 64
MAX_BATCHES = 5000
CHECKPOINT_INTERVAL = 100


# --- Core Influence Computation --------------------------------------------

def compute_token_vector_influence(resid_vector, input_embeds, forward_fn=None):
    """R(t') = sum_{nu,mu} (dy_nu/dx_{mu,t'})^2 for y at the last position.

    If forward_fn is provided, uses torch.autograd.functional.jacobian in a
    single vectorized call (much faster than looping backward() per component).
    """
    from torch.autograd.functional import jacobian

    seq_len = input_embeds.shape[1]
    d_model = resid_vector.shape[0] if resid_vector is not None else None
    device = input_embeds.device

    if forward_fn is not None:
        if d_model is None:
            d_model = forward_fn(input_embeds).shape[0]

        input_flat = input_embeds.detach().clone().requires_grad_(True).reshape(-1)

        def forward_flat(input_flat):
            input_reshaped = input_flat.reshape(1, seq_len, -1)
            return forward_fn(input_reshaped)

        J = jacobian(forward_flat, input_flat)
        input_d_model = input_embeds.shape[-1]
        J_reshaped = J.reshape(d_model, seq_len, input_d_model)
        influence_norms = (J_reshaped ** 2).sum(dim=(0, 2))
        return influence_norms

    if d_model is None:
        raise ValueError("If forward_fn is None, resid_vector must be provided")
    influence_norms = torch.zeros(seq_len, device=device)
    for nu in range(d_model):
        grad_nu = torch.autograd.grad(
            outputs=resid_vector[nu], inputs=input_embeds,
            retain_graph=(nu < d_model - 1), create_graph=False,
            only_inputs=True, allow_unused=False,
        )[0][0]
        influence_norms += (grad_nu ** 2).sum(dim=-1)
    return influence_norms


def process_batch_with_token_influence(model, tokens, layer_idx, preset: Preset):
    """Compute token-vector influence distribution + its entropy for one batch."""
    input_ids = tokens.unsqueeze(0)

    embed_layer = get_embed(model, preset)
    input_embeds = embed_layer(input_ids)
    if isinstance(input_embeds, tuple):
        input_embeds = input_embeds[0]
    input_embeds = input_embeds.detach()
    input_embeds.requires_grad_(True)

    layer = get_layer(model, preset, layer_idx)
    activations = []

    def hook_fn(module, inputs, output):  # noqa: ARG001
        activations.append(output[0] if isinstance(output, tuple) else output)

    handle = layer.register_forward_hook(hook_fn)
    original_embed = get_embed(model, preset)
    dummy = DummyEmbed(input_embeds)
    set_embed(model, preset, dummy)

    try:
        def forward_fn(input_embeds_new):
            local_activations = []

            def local_hook_fn(module, inputs, output):  # noqa: ARG001
                local_activations.append(
                    output[0] if isinstance(output, tuple) else output
                )

            local_handle = layer.register_forward_hook(local_hook_fn)
            try:
                set_embed(model, preset, DummyEmbed(input_embeds_new))
                _ = model(input_ids)
                return local_activations[0][0, -1, :]
            finally:
                local_handle.remove()
                set_embed(model, preset, dummy)

        _ = model(input_ids)
        resid = activations[0]
        last_pos_resid = resid[0, -1, :]

        influence_norms = compute_token_vector_influence(
            resid_vector=last_pos_resid, input_embeds=input_embeds,
            forward_fn=forward_fn,
        )

        R_values = influence_norms.detach().cpu().numpy()
        eps = 1e-12
        Q = (R_values + eps) / (R_values.sum() + eps)
        entropy = scipy.stats.entropy(Q, base=2)
    finally:
        set_embed(model, preset, original_embed)
        handle.remove()

    return R_values, entropy


# --- Standalone main (per-layer, one preset) -------------------------------

def main(preset_name="pythia-70m", layer_idx=3, output_file=None,
         checkpoint_file=None, resume=True, max_batches=None):
    preset = get_preset(preset_name)
    site = site_for(preset, layer_idx)
    max_batches = max_batches if max_batches is not None else MAX_BATCHES

    print(f"\n[INFO] Processing preset={preset.name} site={site}")
    model, tokenizer = load_model(preset, DEVICE)

    from data_loader import load_wikitext_train_text
    text = load_wikitext_train_text()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_tokens = tokens.shape[0]

    if checkpoint_file is None:
        checkpoint_file = Path(f"token_vector_influence_{site}_checkpoint.pt")
    else:
        checkpoint_file = Path(checkpoint_file)

    influence_distributions = []
    entropies = []
    start_idx = 0
    batch_count = 0
    if resume and checkpoint_file.exists():
        ck = torch.load(checkpoint_file, map_location="cpu")
        influence_distributions = ck.get("influence_distributions", [])
        entropies = ck.get("entropies", [])
        batch_count = ck.get("batch_count", 0)
        start_idx = ck.get("start_idx", 0)
        print(f"[INFO] Resuming: batch={batch_count}")

    t0 = time.time()
    for i in range(start_idx, total_tokens - BATCH_SIZE, BATCH_SIZE):
        if batch_count >= max_batches:
            break
        chunk = tokens[i: i + BATCH_SIZE].to(DEVICE)
        try:
            R_values, entropy = process_batch_with_token_influence(
                model, chunk, layer_idx, preset
            )
            influence_distributions.append(R_values)
            entropies.append(entropy)
        except Exception as e:
            print(f"[WARN] batch {batch_count}: {e}")
            continue

        batch_count += 1
        if batch_count % 10 == 0:
            print(f"Processed {batch_count}/{max_batches}...", end="\r")
        if batch_count % CHECKPOINT_INTERVAL == 0:
            torch.save({
                "influence_distributions": influence_distributions,
                "entropies": entropies, "batch_count": batch_count,
                "start_idx": i + BATCH_SIZE,
                "config": {"layer": layer_idx, "site": site,
                           "batch_size": BATCH_SIZE, "preset": preset.name},
            }, checkpoint_file)

    print(f"\nProcessed {batch_count} batches in {time.time()-t0:.1f}s.")
    if not influence_distributions:
        return

    influence_array = np.stack(influence_distributions, axis=0)
    entropies_array = np.array(entropies)
    print(f"[INFO] Mean entropy: {entropies_array.mean():.4f} ± "
          f"{entropies_array.std():.4f} bits")

    output_data = {
        "influence_distributions": influence_array.tolist(),
        "entropies": entropies_array.tolist(),
        "mean_influence": influence_array.mean(axis=0).tolist(),
        "std_influence": influence_array.std(axis=0).tolist(),
        "mean_entropy": float(entropies_array.mean()),
        "std_entropy": float(entropies_array.std()),
        "config": {"preset": preset.name, "layer": layer_idx, "site": site,
                   "total_batches": batch_count, "batch_size": BATCH_SIZE},
    }
    if output_file is None:
        output_file = Path(f"token_vector_influence_{site}.pt")
    torch.save(output_data, Path(output_file))
    print(f"[INFO] Saved results to {output_file}")
    if checkpoint_file.exists():
        checkpoint_file.unlink()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Token-vector influence entropy")
    parser.add_argument("--preset", type=str, default="pythia-70m")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    main(
        preset_name=args.preset, layer_idx=args.layer,
        output_file=args.output_file, checkpoint_file=args.checkpoint_file,
        resume=not args.no_resume, max_batches=args.max_batches,
    )
