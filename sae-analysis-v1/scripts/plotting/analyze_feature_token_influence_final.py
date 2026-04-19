# Analyze which tokens have strong influence on a given feature
# Copy this into a Jupyter notebook cell

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import random

# Configuration - change these as needed
SITE = "resid_out_layer3"  # Change this to desired site/layer
FEATURE_IDX = 100  # Change this to desired feature index
NUM_SAMPLE_BATCHES = 3  # Number of sample batches to process and analyze
TOP_K_TOKENS = 10  # Number of top influential tokens to show per batch

# Load the feature token influence data to understand the feature
print(f"Loading feature influence data for {SITE}...")
data_file = f"feature_token_influence_{SITE}.pt"
data = torch.load(data_file, map_location='cpu', weights_only=False)

feature_influences = data['feature_influences']
config = data.get('config', {})
BATCH_SIZE = config.get('batch_size', 64)
THRESHOLD = config.get('threshold', 0.2)

# Check if feature exists
if FEATURE_IDX not in feature_influences:
    available_features = sorted(feature_influences.keys())
    print(f"ERROR: Feature {FEATURE_IDX} not found in data.")
    print(f"Available features: {available_features[:20]}..." if len(
        available_features) > 20 else f"Available features: {available_features}")
    raise KeyError(f"Feature {FEATURE_IDX} not found")

print(f"Feature {FEATURE_IDX} found in data")
print(f"Batch size: {BATCH_SIZE}, Threshold: {THRESHOLD}")

# Load model and tokenizer to process sample batches
print(f"\nLoading model and tokenizer...")
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
layer_idx = int(SITE.rsplit("layer", 1)[-1])

# Helper functions for SAE


def load_sae(run_dir):
    path = run_dir / "ae.pt"
    if not path.exists():
        raise FileNotFoundError(f"No ae.pt found in {run_dir}")
    return torch.load(path, map_location="cpu")


def get_sae_weights(sd, d_model):
    dec_w = None
    for k, v in sd.items():
        if "decoder.weight" in k or "dec.weight" in k:
            dec_w = v
            break
    if dec_w is None:
        for v in sd.values():
            if v.ndim == 2 and (v.shape[0] == d_model or v.shape[1] == d_model):
                dec_w = v
                break
    if dec_w is None:
        raise ValueError("Could not find decoder weights")
    if dec_w.shape[0] != d_model:
        dec_w = dec_w.T

    enc_w = sd.get("encoder.weight")
    enc_b = sd.get("encoder.bias")
    if enc_w is None:
        enc_w = dec_w.T
    else:
        if enc_w.shape[1] == d_model:
            enc_w = enc_w.T

    if enc_b is None:
        enc_b = torch.zeros(dec_w.shape[1])

    return {
        "enc_w": enc_w,
        "enc_b": enc_b,
        "dec_w": dec_w
    }


def get_feature_activations(resid, sae_weights):
    import torch.nn.functional as F
    x = resid
    enc_w = sae_weights["enc_w"]
    enc_b = sae_weights["enc_b"]
    return F.relu(torch.matmul(x, enc_w) + enc_b)


def compute_influence_for_feature(feature_activation, input_embeds):
    """Compute influence J(t,z|t') = ||∂f_a/∂x_t'||_2^2"""
    if input_embeds.grad is not None:
        input_embeds.grad.zero_()
    feature_activation.backward(retain_graph=True)
    grads = input_embeds.grad
    if grads is None:
        return torch.zeros(input_embeds.shape[1], device=input_embeds.device)
    grads = grads.clone()
    influence_norms = (grads ** 2).sum(dim=-1)
    return influence_norms[0]


# Load SAE weights
base_dir = Path("dictionaries/pythia-70m-deduped") / SITE
run_dir = None
for p in base_dir.iterdir():
    if p.is_dir() and (p / "ae.pt").exists():
        run_dir = p
        break
if not run_dir:
    raise FileNotFoundError(f"No SAE found for {SITE}")

sae_sd = load_sae(run_dir)
d_model = model.config.hidden_size
sae_weights = get_sae_weights(sae_sd, d_model)

# Load text data
DATA_FILE = Path("wikitext-2-train.txt")
if not DATA_FILE.exists():
    raise FileNotFoundError(f"{DATA_FILE} not found")

print(f"Loading text data...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()
all_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
print(f"Loaded {len(all_tokens)} tokens")

# Process sample batches where the feature is activated
print(
    f"\nProcessing sample batches to find where Feature {FEATURE_IDX} is activated...")
max_start = len(all_tokens) - BATCH_SIZE
sample_batches_data = []

# Try to find batches where the feature is activated
attempts = 0
max_attempts = 100
while len(sample_batches_data) < NUM_SAMPLE_BATCHES and attempts < max_attempts:
    start_idx = random.randint(0, max_start)
    batch_tokens = all_tokens[start_idx:start_idx +
                              BATCH_SIZE].unsqueeze(0)  # [1, BATCH_SIZE]

    # Check if feature is activated in this batch
    with torch.no_grad():
        # Get activations
        input_ids = batch_tokens
        layer = model.gpt_neox.layers[layer_idx]
        activations = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations.append(output[0])
            else:
                activations.append(output)

        handle = layer.register_forward_hook(hook_fn)
        _ = model(input_ids)
        handle.remove()

        resid = activations[0]  # [1, seq_len, d_model]
        feats = get_feature_activations(
            resid, sae_weights)  # [1, seq_len, n_latent]
        last_pos_feats = feats[0, -1, :]  # [n_latent]

        activation_value = float(last_pos_feats[FEATURE_IDX].item())

    # If feature is activated, compute influence with gradients (outside no_grad context)
    if activation_value > THRESHOLD:
        # Get embeddings with gradient tracking
        embed_layer = model.gpt_neox.embed_in
        input_embeds = embed_layer(input_ids)
        if isinstance(input_embeds, tuple):
            input_embeds = input_embeds[0]
        input_embeds = input_embeds.detach()
        input_embeds.requires_grad_(True)

           # Recompute with gradients enabled
           # Create dummy embedding layer
           class DummyEmbed(torch.nn.Module):
                def __init__(self, embeds):
                    super().__init__()
                    self.embeds = embeds

                def forward(self, input_ids):
                    return self.embeds

            original_embed = model.gpt_neox.embed_in
            dummy = DummyEmbed(input_embeds)
            model.gpt_neox.embed_in = dummy

            try:
                activations_grad = []

                def hook_fn_grad(module, input, output):
                    if isinstance(output, tuple):
                        activations_grad.append(output[0])
                    else:
                        activations_grad.append(output)

                handle_grad = layer.register_forward_hook(hook_fn_grad)
                _ = model(input_ids)
                handle_grad.remove()

                   resid_grad = activations_grad[0]  # [1, seq_len, d_model]
                # Ensure SAE weights are on the same device as resid_grad
                device = resid_grad.device
                sae_weights_grad = {
                    "enc_w": sae_weights["enc_w"].to(device),
                    "enc_b": sae_weights["enc_b"].to(device),
                    "dec_w": sae_weights["dec_w"].to(device)
                }
                feats_grad = get_feature_activations(
                    resid_grad, sae_weights_grad)  # [1, seq_len, n_latent]
                feat_activation = feats_grad[0, -1, FEATURE_IDX]

                # Verify gradient connection
                if not feat_activation.requires_grad:
                    print(
                        f"  Warning: feat_activation does not require grad, skipping batch")
                    model.gpt_neox.embed_in = original_embed
                    attempts += 1
                    continue

                influence_norms = compute_influence_for_feature(
                    feat_activation, input_embeds)
                influence_dist = influence_norms.detach().cpu().numpy()
            finally:
                model.gpt_neox.embed_in = original_embed

            sample_batches_data.append({
                'start_idx': start_idx,
                'tokens': batch_tokens[0].cpu().numpy(),  # [BATCH_SIZE]
                'influence': influence_dist,  # [BATCH_SIZE]
                'activation': activation_value
            })
            print(
                f"  Found batch at start_idx={start_idx}, activation={activation_value:.4f}")

    attempts += 1

if len(sample_batches_data) == 0:
    print(
        f"Warning: Could not find batches where Feature {FEATURE_IDX} is activated.")
    print("Showing influence patterns from stored data instead...")
    # Fall back to stored data
    feature_data = feature_influences[FEATURE_IDX]
    all_influences = feature_data['all_influences']
    if len(all_influences) > 0:
        batch_idx = random.randint(0, len(all_influences) - 1)
        influence_dist = np.array(all_influences[batch_idx])
        print(f"\nShowing influence from stored batch {batch_idx}:")
        top_k_indices = np.argsort(influence_dist)[-TOP_K_TOKENS:][::-1]
        print(f"{'Position':<10} | {'Influence':<12} | {'Normalized':<12}")
        print("-" * 40)
        total_influence = np.sum(influence_dist)
        for pos_idx in top_k_indices:
            influence_val = influence_dist[pos_idx]
            normalized = influence_val / total_influence if total_influence > 0 else 0
            print(f"{pos_idx:<10} | {influence_val:<12.6f} | {normalized:<12.4%}")
else:
    # Analyze the sample batches
    print(f"\n{'='*80}")
    print(f"Analysis of Feature {FEATURE_IDX} - Tokens with Strong Influence")
    print(f"{'='*80}")
    print(
        f"\nFound {len(sample_batches_data)} batches where feature is activated")
    print(f"Note: Influence shows how each token position affects the feature activation.")
    print(f"Higher influence = stronger effect on the feature at the last token position.\n")

    for batch_data in sample_batches_data:
        start_idx = batch_data['start_idx']
        tokens = batch_data['tokens']
        influence_dist = batch_data['influence']
        activation = batch_data['activation']
        seq_len = len(influence_dist)

        print(f"\n{'='*80}")
        print(
            f"Batch at start_idx={start_idx} (Feature activation: {activation:.4f})")
        print(f"{'='*80}")

        # Get top K token positions by influence
        top_k_indices = np.argsort(influence_dist)[-TOP_K_TOKENS:][::-1]

        print(f"\nTop {len(top_k_indices)} tokens by influence:")
        print(
            f"{'Position':<10} | {'Influence':<12} | {'Normalized':<12} | {'Token Text'}")
        print("-" * 90)

        # Normalize influence for better interpretation
        total_influence = np.sum(influence_dist)

        for pos_idx in top_k_indices:
            influence_val = influence_dist[pos_idx]
            normalized = influence_val / total_influence if total_influence > 0 else 0

            # Decode actual token
            token_id = int(tokens[pos_idx])
            token_text = tokenizer.decode([token_id])
            # Clean up for display
            token_text = repr(token_text.replace(
                '\n', '\\n').replace('\r', '\\r')[:40])

            print(
                f"{pos_idx:<10} | {influence_val:<12.6f} | {normalized:<12.4%} | {token_text}")

        # Show full context around top tokens
        print(f"\nContext around top influential tokens:")
        for pos_idx in top_k_indices[:5]:  # Show top 5
            context_start = max(0, pos_idx - 3)
            context_end = min(seq_len, pos_idx + 4)
            context_tokens = tokens[context_start:context_end]
            context_text = tokenizer.decode(context_tokens.tolist())
            marker_pos = pos_idx - context_start
            # Mark the influential token
            context_parts = context_text.split(' ')
            if marker_pos < len(context_parts):
                context_parts[marker_pos] = f"**{context_parts[marker_pos]}**"
                context_text = ' '.join(context_parts)
            print(f"  Position {pos_idx}: ...{context_text[:100]}...")

        # Visualize influence distribution
        plt.figure(figsize=(12, 5))
        plt.plot(influence_dist, 'b-', linewidth=1.5,
                 alpha=0.7, label='Influence')
        plt.scatter(top_k_indices, influence_dist[top_k_indices],
                    color='red', s=50, zorder=5, label=f'Top {TOP_K_TOKENS} positions')
        plt.xlabel('Token Position in Batch', fontsize=11, fontweight='bold')
        plt.ylabel('Influence Value', fontsize=11, fontweight='bold')
        plt.title(f'Feature {FEATURE_IDX} - Batch at start_idx={start_idx}\nInfluence Distribution (Activation: {activation:.4f})',
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Show influence distribution statistics
        print(f"\nInfluence distribution statistics:")
        print(f"  Total influence: {total_influence:.6f}")
        print(f"  Mean: {influence_dist.mean():.6f}")
        print(
            f"  Max: {influence_dist.max():.6f} (at position {np.argmax(influence_dist)})")
        print(f"  Min: {influence_dist.min():.6f}")
        print(f"  Std: {influence_dist.std():.6f}")

        # Show which positions have the strongest influence (top 3)
        top3_positions = np.argsort(influence_dist)[-3:][::-1]
        print(f"\nTop 3 positions: {top3_positions.tolist()}")
        print(
            f"  These positions contribute {np.sum(influence_dist[top3_positions])/total_influence*100:.1f}% of total influence")

print(f"\n{'='*80}")
print("Analysis complete!")
print(f"{'='*80}")
