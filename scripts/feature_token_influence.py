import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
THRESHOLD = 1.0
BATCH_SIZE = 64  # Smaller batch size for gradient computation
MAX_BATCHES = 5000  # Limit number of batches to process
MIN_FEATURE_ACTIVATIONS = 10  # Minimum activations to include a feature
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N batches

# --- Helper Functions (Copied from existing modules for standalone usage) ---
def load_sae(run_dir):
    path = run_dir / "ae.pt"
    if not path.exists():
        raise FileNotFoundError(f"No ae.pt found in {run_dir}")
    print(f"[INFO] Loading SAE from {path}")
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
    if dec_w is None: raise ValueError("Could not find decoder weights")
    if dec_w.shape[0] != d_model: dec_w = dec_w.T
    
    enc_w = sd.get("encoder.weight")
    enc_b = sd.get("encoder.bias")
    if enc_w is None: enc_w = dec_w.T
    else:
        if enc_w.shape[1] == d_model: enc_w = enc_w.T
    
    if enc_b is None: enc_b = torch.zeros(dec_w.shape[1])
    
    return {
        "enc_w": enc_w.to(DEVICE),
        "enc_b": enc_b.to(DEVICE),
        "dec_w": dec_w.to(DEVICE)
    }

def get_feature_activations(resid, sae_weights):
    x = resid
    enc_w = sae_weights["enc_w"]
    enc_b = sae_weights["enc_b"]
    return F.relu(torch.matmul(x, enc_w) + enc_b)


# --- Core Influence Computation Functions ---

def compute_influence_for_feature(feature_activation, input_embeds):
    """
    Compute the influence matrix I_aμ(t,z|t') = ∂f_a(t,z)/∂x_μ,t'
    
    Args:
        feature_activation: Scalar activation value for feature a at position t
        input_embeds: Input embeddings tensor [1, seq_len, d_model] with gradients enabled
    
    Returns:
        influence_norms: [seq_len] tensor of J(t,z|t') = ||I_aμ(t,z|t')||_2^2
    """
    # Compute gradients
    if input_embeds.grad is not None:
        input_embeds.grad.zero_()
    
    feature_activation.backward(retain_graph=True)
    
    # Get gradient: [1, seq_len, d_model]
    grads = input_embeds.grad
    
    if grads is None:
        # This can happen if the feature doesn't depend on the input (shouldn't happen here)
        return torch.zeros(input_embeds.shape[1], device=input_embeds.device)
        
    grads = grads.clone()
    
    # Compute L2 norm squared for each position: ||∂f_a/∂x_t'||_2^2
    # Sum over d_model dimension
    influence_norms = (grads ** 2).sum(dim=-1)  # [1, seq_len]
    
    return influence_norms[0]  # [seq_len]


def process_batch_with_influence(model, tokenizer, sae_weights, tokens, layer_idx, leading_features, threshold):
    """
    Process a batch of tokens and compute influence for activated leading features.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        sae_weights: SAE weights dictionary
        tokens: Token IDs [batch_size]
        layer_idx: Layer index to analyze
        leading_features: Set of feature indices to track
        threshold: Activation threshold
    
    Returns:
        feature_influences: Dict mapping feature_idx -> influence_distribution [seq_len]
    """
    seq_len = tokens.shape[0]
    
    # Prepare input with gradient tracking
    # We need to replace the embedding layer temporarily to track gradients
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
    # We need to bypass the embedding layer and use our custom embeddings
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
        # Forward pass
        _ = model(input_ids)
        
        # Get the captured activation
        resid = activations[0]  # [1, seq_len, d_model]
        
        # Get feature activations
        feats = get_feature_activations(resid, sae_weights)  # [1, seq_len, n_latent]
        
        # For the last token position, check which leading features are active
        last_pos_feats = feats[0, -1, :]  # [n_latent]
        
        feature_influences = {}
        
        for feat_idx in leading_features:
            if last_pos_feats[feat_idx] > threshold:
                # Compute influence for this feature
                feat_activation = last_pos_feats[feat_idx]
                
                # Compute gradients
                # Pass the full input_embeds tensor which is the leaf
                influence_norms = compute_influence_for_feature(feat_activation, input_embeds)
                
                feature_influences[feat_idx] = influence_norms.detach().cpu().numpy()
        
    finally:
        # Restore original embedding layer
        model.gpt_neox.embed_in = original_embed
        handle.remove()
    
    return feature_influences


# --- Main Analysis ---

def load_checkpoint(checkpoint_file):
    """
    Load checkpoint data from file.
    
    Args:
        checkpoint_file: Path to checkpoint file
    
    Returns:
        checkpoint_data: Dict with influence_data, batch_count, start_idx, config
        or None if file doesn't exist
    """
    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        return None
    
    print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
    data = torch.load(checkpoint_path, map_location="cpu")
    
    # Convert lists back to defaultdict
    influence_data = defaultdict(list)
    if "influence_data" in data:
        for feat_idx, influence_list in data["influence_data"].items():
            influence_data[feat_idx] = influence_list
    
    return {
        "influence_data": influence_data,
        "batch_count": data.get("batch_count", 0),
        "start_idx": data.get("start_idx", 0),
        "config": data.get("config", {})
    }


def save_checkpoint(checkpoint_file, influence_data, batch_count, start_idx, config):
    """
    Save checkpoint data to file.
    
    Args:
        checkpoint_file: Path to checkpoint file
        influence_data: defaultdict mapping feature_idx -> list of influence distributions
        batch_count: Current batch count
        start_idx: Current start index in token sequence
        config: Configuration dict
    """
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert defaultdict to regular dict for serialization
    influence_data_dict = dict(influence_data)
    
    checkpoint_data = {
        "influence_data": influence_data_dict,
        "batch_count": batch_count,
        "start_idx": start_idx,
        "config": config
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path} ({batch_count} batches)")


def main(site=None, sparsity_file=None, output_file=None, checkpoint_file=None, resume=True):
    """
    Main function to compute feature token influence.
    
    Args:
        site: Site string like "resid_out_layer3" (if None, uses SITE from config)
        sparsity_file: Path to feature_sparsity_data.pt (if None, uses default or layer-specific)
        output_file: Output file path (if None, uses layer-specific name)
    """
    # Use provided site or default
    if site is None:
        site = "resid_out_layer3"
    
    base_dir = Path("dictionaries/pythia-70m-deduped") / site
    
    # 1. Setup
    run_dir = None
    for p in base_dir.iterdir():
        if p.is_dir() and (p / "ae.pt").exists():
            run_dir = p
            break
    if not run_dir: 
        raise FileNotFoundError(f"No run directory found in {base_dir}")

    print(f"\n{'='*60}")
    print(f"[INFO] Processing {site}")
    print(f"{'='*60}")

    print("[INFO] Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    d_model = model.config.hidden_size
    
    sae_sd = load_sae(run_dir)
    sae_weights = get_sae_weights(sae_sd, d_model)
    n_latent = sae_weights["dec_w"].shape[1]
    
    # 2. Load leading features from feature_sparsity_data.pt
    if sparsity_file is None:
        # Try layer-specific file first, then fall back to default
        sparsity_file = Path(f"feature_sparsity_data_{site}.pt")
        if not sparsity_file.exists():
            sparsity_file = Path("feature_sparsity_data.pt")
    
    if not sparsity_file.exists():
        print(f"[ERROR] {sparsity_file} not found. Please run feature_sparsity.py first.")
        sys.exit(1)
    
    print(f"[INFO] Loading feature sparsity data from {sparsity_file}...")
    sparsity_data = torch.load(sparsity_file)
    frequencies = sparsity_data["frequencies"]
    
    # Select leading features (those with sufficient activation)
    MIN_FREQ = 0.001  # At least 0.1% activation frequency
    MAX_FEATURES = 500  # Maximum number of features to select
    leading_mask = frequencies > MIN_FREQ
    candidate_indices = torch.nonzero(leading_mask).squeeze(-1)
    
    # If too many features, take top N by frequency
    if len(candidate_indices) > MAX_FEATURES:
        # Get frequencies for candidate features and select top N by frequency
        candidate_freqs = frequencies[candidate_indices]
        _, top_indices = torch.topk(candidate_freqs, MAX_FEATURES, largest=True)
        leading_features = set(candidate_indices[top_indices].tolist())
        print(f"[INFO] Found {len(candidate_indices)} features above threshold, selecting top {MAX_FEATURES} by frequency")
    else:
        leading_features = set(candidate_indices.tolist())
    
    print(f"[INFO] Selected {len(leading_features)} leading features (freq > {MIN_FREQ:.1%})")
    
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
        checkpoint_file = Path(f"feature_token_influence_{site}_checkpoint.pt")
    else:
        checkpoint_file = Path(checkpoint_file)
    
    # Storage for influence distributions
    # feature_idx -> list of influence distributions (each is [seq_len] array)
    influence_data = defaultdict(list)
    
    # Try to resume from checkpoint
    start_idx = 0
    batch_count = 0
    if resume and checkpoint_file.exists():
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            influence_data = checkpoint_data["influence_data"]
            batch_count = checkpoint_data["batch_count"]
            start_idx = checkpoint_data["start_idx"]
            print(f"[INFO] Resuming from checkpoint: {batch_count} batches already processed")
            print(f"[INFO] Resuming from token index: {start_idx}")
            print(f"[INFO] Features with data: {len(influence_data)}")
    
    print(f"[INFO] Computing token influence (Threshold > {THRESHOLD})...")
    print(f"[INFO] Processing up to {MAX_BATCHES} batches of size {BATCH_SIZE}...")
    
    import time
    start_time = time.time()
    
    # Process batches starting from start_idx
    for i in range(start_idx, total_tokens - BATCH_SIZE, BATCH_SIZE):
        if batch_count >= MAX_BATCHES:
            break
            
        chunk = tokens[i : i + BATCH_SIZE].to(DEVICE)  # [BATCH_SIZE]
        
        # Compute influence for this batch
        try:
            batch_influences = process_batch_with_influence(
                model, tokenizer, sae_weights, chunk, layer_idx, leading_features, THRESHOLD
            )
            
            # Store results
            for feat_idx, influence_dist in batch_influences.items():
                influence_data[feat_idx].append(influence_dist)
            
        except Exception as e:
            print(f"[WARN] Error processing batch {batch_count}: {e}")
            continue
        
        batch_count += 1
        
        # Progress
        if batch_count % 10 == 0:
            print(f"Processed {batch_count}/{MAX_BATCHES} batches...", end="\r")
        
        # Save checkpoint periodically
        if batch_count % CHECKPOINT_INTERVAL == 0:
            config = {
                "threshold": THRESHOLD,
                "layer": layer_idx,
                "site": site,
                "batch_size": BATCH_SIZE,
                "min_freq": MIN_FREQ,  # Defined earlier in main()
            }
            save_checkpoint(checkpoint_file, influence_data, batch_count, 
                          i + BATCH_SIZE, config)
    
    print(f"\nProcessed {batch_count} batches. Done.")
    
    elapsed = time.time() - start_time
    print(f"[INFO] Processing took {elapsed:.2f}s ({batch_count/elapsed:.1f} batches/s)")
    
    # 5. Aggregate and Save Results
    print("[INFO] Aggregating influence distributions...")
    
    aggregated_data = {}
    for feat_idx, influence_list in influence_data.items():
        if len(influence_list) >= MIN_FEATURE_ACTIVATIONS:
            # Stack and compute statistics
            influence_array = np.stack(influence_list, axis=0)  # [num_samples, seq_len]
            
            aggregated_data[feat_idx] = {
                "mean_influence": influence_array.mean(axis=0).tolist(),  # [seq_len]
                "std_influence": influence_array.std(axis=0).tolist(),    # [seq_len]
                "all_influences": influence_array.tolist(),               # [num_samples, seq_len]
                "num_samples": len(influence_list),
            }

    print(f"[INFO] Aggregated data for {len(aggregated_data)} features")
    
    # Save results
    output_data = {
        "feature_influences": aggregated_data,
        "config": {
            "threshold": THRESHOLD,
            "layer": layer_idx,
            "site": site,
            "total_batches": batch_count,
            "batch_size": BATCH_SIZE,
            "min_freq": MIN_FREQ,
        }
    }
    
    if output_file is None:
        output_file = Path(f"feature_token_influence_{site}.pt")
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
    print("Token Influence Analysis Summary")
    print("-" * 60)
    print(f"Site: {site}")
    print(f"Features analyzed: {len(aggregated_data)}")
    print(f"Batches processed: {batch_count}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Show top features by number of samples
    if aggregated_data:
        sorted_features = sorted(
            aggregated_data.items(),
            key=lambda x: x[1]["num_samples"],
            reverse=True
        )[:10]
        
        print(f"\nTop 10 features by activation count:")
        print(f"{'Feature':<10} | {'Samples':<10} | {'Mean Total Influence':<20}")
        for feat_idx, data in sorted_features:
            total_influence = sum(data["mean_influence"])
            print(f"{feat_idx:<10} | {data['num_samples']:<10} | {total_influence:<20.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute feature token influence")
    parser.add_argument("--site", type=str, default=None, help="Site like 'resid_out_layer3'")
    parser.add_argument("--sparsity-file", type=str, default=None, 
                       help="Path to feature_sparsity_data.pt")
    parser.add_argument("--output-file", type=str, default=None, help="Output file path")
    parser.add_argument("--checkpoint-file", type=str, default=None, 
                       help="Checkpoint file path (default: auto-generated)")
    parser.add_argument("--no-resume", action="store_true", 
                       help="Don't resume from checkpoint even if it exists")
    
    args = parser.parse_args()
    
    # Option 1: Run for a single layer (from command line)
    if args.site:
        try:
            main(site=args.site, sparsity_file=args.sparsity_file, 
                 output_file=args.output_file, checkpoint_file=args.checkpoint_file,
                 resume=not args.no_resume)
        except Exception as e:
            print(f"[ERROR] Failed to process {args.site}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Option 2: Run for multiple layers (default behavior)
        sites = [
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