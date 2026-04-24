import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import scipy.stats

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64  # Batch size for processing
MAX_BATCHES = 5000  # Limit number of batches to process
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N batches

# --- Core Influence Computation Functions ---


def compute_token_vector_influence(resid_vector, input_embeds, forward_fn=None):
    """
    Compute the influence matrix D_νμ(t,z|t') = ∂y_ν(t,z)/∂x_μ(t')

    According to the note:
    - D_νμ(t,z|t') = ∂y_ν(t,z)/∂x_μ(t')
    - R(t,z|t') = ||D_νμ(t,z|t')||^2 = tr(DD^†)

    Uses a single vectorized jacobian computation via torch.autograd.functional.jacobian.

    Args:
        resid_vector: Internal token vector at position t [d_model] (the last token position)
                     If forward_fn is provided, this can be None (will be computed)
        input_embeds: Input embeddings tensor [1, seq_len, d_model] with gradients enabled
        forward_fn: Optional function that takes input_embeds and returns resid_vector [d_model]
                   If None, uses the existing computation graph (less efficient)

    Returns:
        influence_norms: [seq_len] tensor of R(t,z|t') = ||D_νμ(t,z|t')||^2
    """
    from torch.autograd.functional import jacobian

    seq_len = input_embeds.shape[1]
    d_model = resid_vector.shape[0] if resid_vector is not None else None
    device = input_embeds.device

    # If forward function is provided, use functional jacobian (most efficient)
    if forward_fn is not None:
        # Get d_model from forward function output if not provided
        if d_model is None:
            test_output = forward_fn(input_embeds)
            d_model = test_output.shape[0]

        # Flatten input for jacobian: [1, seq_len, d_model] -> [seq_len * d_model]
        input_flat = input_embeds.detach().clone().requires_grad_(True).reshape(-1)

        def forward_flat(input_flat):
            # Reshape back to [1, seq_len, d_model]
            input_reshaped = input_flat.reshape(1, seq_len, -1)
            # Call forward function and extract last position
            # Should return [d_model] for last position
            resid = forward_fn(input_reshaped)
            return resid

        # Compute jacobian: J[ν, μ*d_model + α] = ∂y_ν/∂x_{μ*d_model + α}
        # Shape: [d_model, seq_len * d_model]
        J = jacobian(forward_flat, input_flat)

        # Get the actual d_model from input (in case it differs)
        input_d_model = input_embeds.shape[-1]

        # Reshape jacobian: [d_model, seq_len, d_model]
        # The second dimension (seq_len * d_model) gets reshaped to [seq_len, d_model]
        J_reshaped = J.reshape(d_model, seq_len, input_d_model)

        # Compute influence: for each position μ, sum over ν and α of |J[ν, μ, α]|²
        # J[ν, μ, :] is ∂y_ν/∂x_μ (vector of length d_model)
        # |∂y_ν/∂x_μ|² = sum_α J[ν, μ, α]²
        # Sum over ν: influence_norms[μ] = sum_ν |∂y_ν/∂x_μ|²
        influence_norms = (J_reshaped ** 2).sum(dim=(0, 2))  # [seq_len]

        return influence_norms

    # Fallback: use existing computation graph (less efficient but works with current structure)
    # This computes gradients in a loop but is still faster than multiple backward() calls
    if d_model is None:
        raise ValueError(
            "If forward_fn is None, resid_vector must be provided")

    # We want: ∑_{μν} |∂y_ν/∂x_μ|² = ∑_μ ∑_ν |∂y_ν/∂x_μ|²
    influence_norms = torch.zeros(seq_len, device=device)

    # Compute all gradients using grad (more efficient than backward)
    # This is still a loop, but grad() is more memory efficient than backward()
    for nu in range(d_model):
        grad_nu = torch.autograd.grad(
            outputs=resid_vector[nu],
            inputs=input_embeds,
            retain_graph=(nu < d_model - 1),
            create_graph=False,
            only_inputs=True,
            allow_unused=False
        )[0][0]  # [seq_len, d_model] - gradient for component nu

        # grad_nu[μ, :] is ∂y_ν/∂x_μ (vector)
        # |∂y_ν/∂x_μ|² = sum over components of x_μ
        influence_norms += (grad_nu ** 2).sum(dim=-1)  # [seq_len]

    return influence_norms  # [seq_len]


def process_batch_with_token_influence(model, tokenizer, tokens, layer_idx):
    """
    Process a batch of tokens and compute influence for the internal token vector.

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        tokens: Token IDs [batch_size]
        layer_idx: Layer index to analyze

    Returns:
        influence_distribution: [seq_len] array of R(t,z|t') values
        entropy: Scalar entropy value for this batch
    """
    seq_len = tokens.shape[0]

    # Prepare input with gradient tracking
    input_ids = tokens.unsqueeze(0)  # [1, seq_len]

    # Get embeddings with gradient tracking
    embed_layer = model.gpt_neox.embed_in

    # Get initial embeddings and detach to make them leaf tensors
    input_embeds = embed_layer(input_ids)
    if isinstance(input_embeds, tuple):
        input_embeds = input_embeds[0]

    input_embeds = input_embeds.detach()
    input_embeds.requires_grad_(True)

    # Hook to capture layer output
    layer = model.gpt_neox.layers[layer_idx]
    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0])
        else:
            activations.append(output)

    handle = layer.register_forward_hook(hook_fn)

    # Forward pass using custom embeddings
    # Save original embedding layer
    original_embed = model.gpt_neox.embed_in

    # Create a dummy embedding layer that returns our tracked embeddings
    class DummyEmbed(torch.nn.Module):
        def __init__(self, embeds):
            super().__init__()
            self.embeds = embeds

        def forward(self, input_ids):
            return self.embeds

    dummy = DummyEmbed(input_embeds)
    model.gpt_neox.embed_in = dummy

    try:
        # Create forward function for efficient jacobian computation
        # This function takes input_embeds and returns resid_vector for last position
        # We need a local activations list for each call
        def forward_fn(input_embeds_new):
            # Create a fresh activations list for this call
            local_activations = []

            def local_hook_fn(module, input, output):
                if isinstance(output, tuple):
                    local_activations.append(output[0])
                else:
                    local_activations.append(output)

            # Register hook temporarily
            local_handle = layer.register_forward_hook(local_hook_fn)

            try:
                # Replace embeddings
                dummy_new = DummyEmbed(input_embeds_new)
                model.gpt_neox.embed_in = dummy_new

                # Forward pass
                _ = model(input_ids)

                # Get captured activation
                resid = local_activations[0]  # [1, seq_len, d_model]
                last_pos_resid = resid[0, -1, :]  # [d_model]

                return last_pos_resid
            finally:
                local_handle.remove()
                # Restore original dummy
                model.gpt_neox.embed_in = dummy

        # Do initial forward pass to get resid_vector for reference
        _ = model(input_ids)
        resid = activations[0]  # [1, seq_len, d_model]
        last_pos_resid = resid[0, -1, :]  # [d_model]

        # Compute influence using efficient jacobian computation
        influence_norms = compute_token_vector_influence(
            resid_vector=last_pos_resid,  # Used for shape info
            input_embeds=input_embeds,
            forward_fn=forward_fn
        )

        # Convert to numpy
        R_values = influence_norms.detach().cpu().numpy()  # [seq_len]

        # Normalize to get probability distribution Q
        # Q_{t,z}(t') = R(t,z|t') / sum_{t'≤t} R(t,z|t')
        eps = 1e-12
        R_sum = np.sum(R_values) + eps
        Q = (R_values + eps) / R_sum

        # Compute entropy
        entropy = scipy.stats.entropy(Q, base=2)

    finally:
        # Restore original embedding layer
        model.gpt_neox.embed_in = original_embed
        handle.remove()

    return R_values, entropy


# --- Main Analysis ---

def load_checkpoint(checkpoint_file):
    """
    Load checkpoint data from file.

    Args:
        checkpoint_file: Path to checkpoint file

    Returns:
        checkpoint_data: Dict with influence_distributions, entropies, batch_count, start_idx
        or None if file doesn't exist
    """
    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        return None

    print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
    data = torch.load(checkpoint_path, map_location="cpu")

    return {
        "influence_distributions": data.get("influence_distributions", []),
        "entropies": data.get("entropies", []),
        "batch_count": data.get("batch_count", 0),
        "start_idx": data.get("start_idx", 0),
        "config": data.get("config", {})
    }


def save_checkpoint(checkpoint_file, influence_distributions, entropies, batch_count, start_idx, config):
    """
    Save checkpoint data to file.

    Args:
        checkpoint_file: Path to checkpoint file
        influence_distributions: List of influence distributions
        entropies: List of entropy values
        batch_count: Current batch count
        start_idx: Current start index in token sequence
        config: Configuration dict
    """
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "influence_distributions": influence_distributions,
        "entropies": entropies,
        "batch_count": batch_count,
        "start_idx": start_idx,
        "config": config
    }

    torch.save(checkpoint_data, checkpoint_path)
    print(
        f"[INFO] Checkpoint saved: {checkpoint_path} ({batch_count} batches)")


def main(site=None, output_file=None, checkpoint_file=None, resume=True):
    """
    Main function to compute token vector influence and entropy.

    Args:
        site: Site string like "resid_out_layer3" (if None, uses default)
        output_file: Output file path (if None, uses layer-specific name)
        checkpoint_file: Checkpoint file path for saving/loading progress
        resume: If True, attempt to resume from checkpoint if it exists
    """
    # Use provided site or default
    if site is None:
        site = "resid_out_layer3"

    print(f"\n{'='*60}")
    print(f"[INFO] Processing {site}")
    print(f"{'='*60}")

    print("[INFO] Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Prepare Data
    DATA_FILE = Path("wikitext-2-train.txt")
    if DATA_FILE.exists():
        print(f"[INFO] Loading data from {DATA_FILE}...")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("[ERROR] wikitext-2-train.txt not found.")
        sys.exit(1)

    print(f"[INFO] Tokenizing text...")
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]  # [Seq]
    total_tokens = tokens.shape[0]
    print(f"[INFO] Total tokens available: {total_tokens}")

    # 4. Process in Batches
    layer_idx = int(site.rsplit("layer", 1)[-1])

    # Setup checkpoint file
    if checkpoint_file is None:
        checkpoint_file = Path(f"token_vector_influence_{site}_checkpoint.pt")
    else:
        checkpoint_file = Path(checkpoint_file)

    # Storage for influence distributions and entropies
    influence_distributions = []  # List of [seq_len] arrays
    entropies = []  # List of scalar entropy values

    # Try to resume from checkpoint
    start_idx = 0
    batch_count = 0
    if resume and checkpoint_file.exists():
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            influence_distributions = checkpoint_data["influence_distributions"]
            entropies = checkpoint_data["entropies"]
            batch_count = checkpoint_data["batch_count"]
            start_idx = checkpoint_data["start_idx"]
            print(
                f"[INFO] Resuming from checkpoint: {batch_count} batches already processed")
            print(f"[INFO] Resuming from token index: {start_idx}")

    print(f"[INFO] Computing token vector influence for layer {layer_idx}...")
    print(
        f"[INFO] Processing up to {MAX_BATCHES} batches of size {BATCH_SIZE}...")

    import time
    start_time = time.time()

    # Process batches starting from start_idx
    for i in range(start_idx, total_tokens - BATCH_SIZE, BATCH_SIZE):
        if batch_count >= MAX_BATCHES:
            break

        chunk = tokens[i: i + BATCH_SIZE].to(DEVICE)  # [BATCH_SIZE]

        # Compute influence for this batch
        try:
            R_values, entropy = process_batch_with_token_influence(
                model, tokenizer, chunk, layer_idx
            )

            # Store results
            influence_distributions.append(R_values)
            entropies.append(entropy)

        except Exception as e:
            print(f"[WARN] Error processing batch {batch_count}: {e}")
            continue

        batch_count += 1

        # Progress
        if batch_count % 10 == 0:
            print(
                f"Processed {batch_count}/{MAX_BATCHES} batches...", end="\r")

        # Save checkpoint periodically
        if batch_count % CHECKPOINT_INTERVAL == 0:
            config = {
                "layer": layer_idx,
                "site": site,
                "batch_size": BATCH_SIZE,
            }
            save_checkpoint(checkpoint_file, influence_distributions, entropies,
                            batch_count, i + BATCH_SIZE, config)

    print(f"\nProcessed {batch_count} batches. Done.")

    elapsed = time.time() - start_time
    print(
        f"[INFO] Processing took {elapsed:.2f}s ({batch_count/elapsed:.1f} batches/s)")

    if len(influence_distributions) == 0:
        print("[ERROR] No batches processed successfully.")
        return

    # 5. Aggregate and Save Results
    print("[INFO] Aggregating results...")

    # Stack influence distributions
    # [num_batches, seq_len]
    influence_array = np.stack(influence_distributions, axis=0)
    entropies_array = np.array(entropies)  # [num_batches]

    # Compute statistics
    mean_influence = influence_array.mean(axis=0).tolist()  # [seq_len]
    std_influence = influence_array.std(axis=0).tolist()    # [seq_len]
    mean_entropy = float(np.mean(entropies_array))
    std_entropy = float(np.std(entropies_array))

    print(f"[INFO] Processed {len(influence_distributions)} batches")
    print(f"[INFO] Mean entropy: {mean_entropy:.4f} ± {std_entropy:.4f} bits")

    # Save results
    output_data = {
        # [num_batches, seq_len]
        "influence_distributions": influence_array.tolist(),
        "entropies": entropies_array.tolist(),  # [num_batches]
        "mean_influence": mean_influence,  # [seq_len]
        "std_influence": std_influence,  # [seq_len]
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "config": {
            "layer": layer_idx,
            "site": site,
            "total_batches": batch_count,
            "batch_size": BATCH_SIZE,
        }
    }

    if output_file is None:
        output_file = Path(f"token_vector_influence_{site}.pt")
    else:
        output_file = Path(output_file)

    torch.save(output_data, output_file)
    print(f"[INFO] Saved results to {output_file}")

    # Remove checkpoint file after successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"[INFO] Removed checkpoint file: {checkpoint_file}")

    # Print summary statistics
    print("\n" + "-" * 60)
    print("Token Vector Influence Analysis Summary")
    print("-" * 60)
    print(f"Site: {site}")
    print(f"Batches processed: {batch_count}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Entropy: {mean_entropy:.4f} ± {std_entropy:.4f} bits")
    print(
        f"Entropy range: {np.min(entropies_array):.4f} to {np.max(entropies_array):.4f} bits")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute token vector influence and entropy")
    parser.add_argument("--site", type=str, default=None,
                        help="Site like 'resid_out_layer3'")
    parser.add_argument("--output-file", type=str,
                        default=None, help="Output file path")
    parser.add_argument("--checkpoint-file", type=str, default=None,
                        help="Checkpoint file path (default: auto-generated)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from checkpoint even if it exists")

    args = parser.parse_args()

    # Option 1: Run for a single layer (from command line)
    if args.site:
        try:
            main(site=args.site, output_file=args.output_file,
                 checkpoint_file=args.checkpoint_file, resume=not args.no_resume)
        except Exception as e:
            print(f"[ERROR] Failed to process {args.site}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Option 2: Run for multiple layers (default behavior)
        sites = [
            "resid_out_layer0",
            "resid_out_layer1",
            "resid_out_layer2",
            "resid_out_layer3",
            "resid_out_layer4",
            "resid_out_layer5",
        ]

        for site in sites:
            try:
                main(site=site)
            except Exception as e:
                print(f"[ERROR] Failed to process {site}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "="*60)
        print("All layers processed!")
        print("="*60)
