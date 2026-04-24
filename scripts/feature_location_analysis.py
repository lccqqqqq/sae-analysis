import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
from collections import Counter
import matplotlib.pyplot as plt
import csv

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
SITE = "resid_out_layer3"
BASE_DIR = Path("dictionaries/pythia-70m-deduped") / SITE
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
THRESHOLD = 1.0
SAMPLE_TEXT_LEN = 2000 # Number of tokens to process
MAX_RECORDS_PER_FEATURE = 10000 # Limit records per feature to avoid memory issues

# --- Helper Functions (Copied from sae_visualizer.py for standalone usage) ---
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


# --- Main Analysis ---

def main():
    # 1. Setup
    run_dir = None
    for p in BASE_DIR.iterdir():
        if p.is_dir() and (p / "ae.pt").exists():
            run_dir = p
            break
    if not run_dir: raise FileNotFoundError("No run directory found")

    print("[INFO] Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    d_model = model.config.hidden_size
    
    sae_sd = load_sae(run_dir)
    sae_weights = get_sae_weights(sae_sd, d_model)
    n_latent = sae_weights["dec_w"].shape[1]
    
    # 2. Prepare Data
    DATA_FILE = Path("wikitext-2-train.txt")
    if DATA_FILE.exists():
        print(f"[INFO] Loading data from {DATA_FILE}...")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("[WARN] wikitext-2-train.txt not found, using dummy text.")
        text = """
        Quantum mechanics is a fundamental theory in physics that describes the behavior of nature at and below the scale of atoms. 
        """ * 50

    print(f"[INFO] Tokenizing text...")
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0] # [Seq]
    total_tokens = tokens.shape[0]
    print(f"[INFO] Total tokens available: {total_tokens}")
    
    # Limit to reasonable amount if needed, but let's try to process up to 100k
    MAX_TOKENS = 1000000
    if total_tokens > MAX_TOKENS:
        print(f"[INFO] Limiting to first {MAX_TOKENS} tokens.")
        tokens = tokens[:MAX_TOKENS]
        total_tokens = MAX_TOKENS
    
    
    # 3. Process in Batches
    BATCH_SIZE = 128
    
    feature_counts = torch.zeros(n_latent, device=DEVICE)
    # NEW: Store activation records with position information
    # Each element is a list of dicts: {'token_id': int, 'abs_pos': int, 'batch_pos': int}
    feature_activations = [[] for _ in range(n_latent)]
    feature_token_counts = [Counter() for _ in range(n_latent)] # Keep for compatibility
    
    layer_idx = int(SITE.rsplit("layer", 1)[-1])
    layer = model.gpt_neox.layers[layer_idx]
    
    print(f"[INFO] Computing feature activations (Threshold > {THRESHOLD})...")
    print(f"[INFO] Max records per feature: {MAX_RECORDS_PER_FEATURE}")
    
    activations = []
    def hook_fn(module, input, output):
        if isinstance(output, tuple): activations.append(output[0].detach())
        else: activations.append(output.detach())
    handle = layer.register_forward_hook(hook_fn)
    
    import time
    start_time = time.time()
    
    for batch_idx, i in enumerate(range(0, total_tokens, BATCH_SIZE)):
        chunk = tokens[i : i + BATCH_SIZE].unsqueeze(0).to(DEVICE) # [1, Chunk]
        batch_start_pos = i  # Absolute position where this batch starts
        
        # Clear previous activations
        activations = []
        
        with torch.no_grad():
            model(chunk)
            
        resid = activations[0] # [1, Chunk, d_model]
        feats = get_feature_activations(resid, sae_weights) # [1, Chunk, n_latent]
        
        # Count
        active_mask = (feats > THRESHOLD) # [1, Chunk, n_latent]
        feature_counts += active_mask.float().sum(dim=(0, 1))
        
        # Track Tokens AND Positions
        # Get indices of active features: (batch, seq, feat_idx)
        # We only have batch size 1 here, so we can ignore dim 0
        # active_indices shape: [N_active, 3]
        active_indices = torch.nonzero(active_mask)
        
        if active_indices.shape[0] > 0:
            # Extract sequences and feature indices
            seq_idxs = active_indices[:, 1]
            feat_idxs = active_indices[:, 2]
            
            # Get the actual token IDs for these activations
            # chunk is [1, Chunk], so chunk[0, seq_idxs] gives the tokens
            active_token_ids = chunk[0, seq_idxs]
            
            # Move to CPU for processing
            feat_idxs_cpu = feat_idxs.cpu().tolist()
            token_ids_cpu = active_token_ids.cpu().tolist()
            seq_idxs_cpu = seq_idxs.cpu().tolist()
            
            for f_idx, t_id, batch_pos in zip(feat_idxs_cpu, token_ids_cpu, seq_idxs_cpu):
                # Update token counter (for compatibility)
                feature_token_counts[f_idx][t_id] += 1
                
                # NEW: Record position information
                # Only store if we haven't exceeded the limit
                if len(feature_activations[f_idx]) < MAX_RECORDS_PER_FEATURE:
                    abs_pos = batch_start_pos + batch_pos
                    feature_activations[f_idx].append({
                        'token_id': t_id,
                        'abs_pos': abs_pos,
                        'batch_pos': batch_pos
                    })

        # Progress
        if batch_idx % 50 == 0:
            print(f"Processed {i}/{total_tokens} tokens...", end="\r")
            
    print(f"Processed {total_tokens}/{total_tokens} tokens. Done.")
    handle.remove()
    
    elapsed = time.time() - start_time
    print(f"[INFO] Processing took {elapsed:.2f}s ({total_tokens/elapsed:.1f} tokens/s)")
    
    # Check if any features hit the record limit
    truncated_features = sum(1 for f in feature_activations if len(f) >= MAX_RECORDS_PER_FEATURE)
    if truncated_features > 0:
        print(f"[WARN] {truncated_features} features hit the {MAX_RECORDS_PER_FEATURE} record limit and were truncated.")
    
    # Metrics
    dead_features = (feature_counts == 0).sum().item()
    frequencies = feature_counts / total_tokens
    
    print("-" * 60)
    print(f"Analysis Results (Sample size: {total_tokens} tokens)")
    print("-" * 60)
    print(f"Total Features: {n_latent}")
    print(f"Dead Features (0 activations): {dead_features} ({dead_features/n_latent:.2%})")
    
    # Top active features
    top_k = 10
    top_vals, top_idxs = torch.topk(frequencies, k=top_k)
    print(f"\nTop {top_k} Most Frequent Features:")
    print(f"{'Feature Idx':<12} | {'Freq':<10} | {'Count':<10} | {'Records':<10} | {'Top Tokens'}")
    for val, idx in zip(top_vals, top_idxs):
        f_idx = idx.item()
        # Get top 3 tokens for this feature
        top_tokens = feature_token_counts[f_idx].most_common(3)
        top_tokens_str = ", ".join([f"{tokenizer.decode([t]).strip()}({c})" for t, c in top_tokens])
        num_records = len(feature_activations[f_idx])
        print(f"{f_idx:<12} | {val.item():.2%}    | {int(val.item() * total_tokens):<10} | {num_records:<10} | {top_tokens_str}")

    # Histogram buckets
    buckets = [0.001, 0.01, 0.05, 0.1] # 0.1%, 1%, 5%, 10%
    print(f"\nFrequency Distribution:")
    for b in buckets:
        count = (frequencies > b).sum().item()
        print(f"Features active > {b:.1%} of time: {count}")

    # Plot Histogram
    print("\n[INFO] Plotting histogram...")
    
    # Filter out dead features for log plot to avoid -inf
    active_freqs = frequencies[frequencies > 0].cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(active_freqs, bins=200, log=True, color='skyblue', edgecolor='black')
    plt.title(f"Feature Activation Frequency Distribution (Threshold > {THRESHOLD})")
    plt.xlabel("Frequency (Activation Probability)")
    plt.ylabel("Count of Features (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    out_file = "location_histogram.png"
    plt.savefig(out_file)
    print(f"[INFO] Histogram saved to {out_file}")

    # Save Data
    print("\n[INFO] Saving data to files...")
    
    # 1. Save as PyTorch tensor dictionary (easy for notebooks)
    data = {
        "feature_counts": feature_counts.cpu(),
        "frequencies": frequencies.cpu(),
        "total_tokens": total_tokens,
        "threshold": THRESHOLD,
        "feature_token_counts": feature_token_counts, # List of Counters
        "feature_activations": feature_activations,  # NEW: List of position records
        "max_records_per_feature": MAX_RECORDS_PER_FEATURE
    }
    torch.save(data, "feature_location_data.pt")
    print("[INFO] Saved 'feature_location_data.pt'")
    
    # 2. Save as CSV (readable)
    # Columns: feature_idx, count, frequency, num_records, top_tokens, position_stats
    with open("feature_location.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_idx", "count", "frequency", "num_records", "top_tokens", "avg_position", "position_std"])
        for idx, (cnt, freq) in enumerate(zip(feature_counts.tolist(), frequencies.tolist())):
            # Get top 5 tokens
            top_t = feature_token_counts[idx].most_common(5)
            # Format: "token1(count), token2(count)"
            # Escape quotes in tokens
            top_t_str = ", ".join([f"{repr(tokenizer.decode([t]))}({c})" for t, c in top_t])
            
            # Calculate position statistics
            num_records = len(feature_activations[idx])
            if num_records > 0:
                positions = [rec['abs_pos'] for rec in feature_activations[idx]]
                avg_pos = sum(positions) / len(positions)
                # Calculate standard deviation
                if len(positions) > 1:
                    variance = sum((p - avg_pos) ** 2 for p in positions) / len(positions)
                    std_pos = variance ** 0.5
                else:
                    std_pos = 0.0
            else:
                avg_pos = 0.0
                std_pos = 0.0
            
            writer.writerow([idx, int(cnt), f"{freq:.6f}", num_records, top_t_str, f"{avg_pos:.2f}", f"{std_pos:.2f}"])
    print("[INFO] Saved 'feature_location.csv'")
    
    print("\n[INFO] Analysis complete!")
    print(f"[INFO] Data saved to 'feature_location_data.pt' and 'feature_location.csv'")
    print(f"[INFO] Original files 'feature_sparsity_data.pt' and 'feature_sparsity.csv' remain unchanged.")
    
if __name__ == "__main__":
    main()
