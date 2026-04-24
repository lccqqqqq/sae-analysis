"""
Compare entropy between feature influences and token vector influences for multiple layers.

This module is an optimized version that processes multiple layers in a single pass,
reusing the model, tokenizer, and data loading to improve efficiency.

For each batch:
1. Loads model, tokenizer, and data once
2. Loads SAE weights for all layers once
3. Processes the same batch across all layers sequentially
4. Saves results separately for each layer (compatible with compare_entropies.py)

Usage:
    python compare_entropies_multi_layer.py --layers 0 1 2 3 4 5 --num-batches 10
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
import numpy as np
import scipy.stats
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import random
import json
from datetime import datetime
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

# Import functions from existing modules
from feature_token_influence import (
    load_sae, get_sae_weights, get_feature_activations,
    process_batch_with_influence
)

THRESHOLD = 0.2

from token_vector_influence import (
    process_batch_with_token_influence
)

# Import plotting and computation functions from compare_entropies
from compare_entropies import (
    compute_feature_entropy,
    normalize_influence,
    plot_batch_comparison,
    compare_entropies_for_batch
)

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64
NUM_BATCHES = 10  # Default number of batches for comparison


def load_all_sae_weights(layers, d_model):
    """
    Load SAE weights for all specified layers.
    
    Args:
        layers: List of layer indices (e.g., [0, 1, 2, 3, 4, 5])
        d_model: Model hidden dimension
    
    Returns:
        sae_weights_dict: Dict mapping layer_idx -> sae_weights dict
    """
    sae_weights_dict = {}
    base_dir = Path("dictionaries/pythia-70m-deduped")
    
    for layer_idx in layers:
        site = f"resid_out_layer{layer_idx}"
        layer_dir = base_dir / site
        
        # Find run directory
        run_dir = None
        for p in layer_dir.iterdir():
            if p.is_dir() and (p / "ae.pt").exists():
                run_dir = p
                break
        
        if not run_dir:
            raise FileNotFoundError(f"No run directory found in {layer_dir}")
        
        print(f"[INFO] Loading SAE for {site}...")
        sae_sd = load_sae(run_dir)
        sae_weights = get_sae_weights(sae_sd, d_model)
        sae_weights_dict[layer_idx] = sae_weights
    
    return sae_weights_dict


def process_batch_for_all_layers(
    model, tokenizer, sae_weights_dict, tokens, layers, all_features_dict
):
    """
    Process a single batch for all layers.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        sae_weights_dict: Dict mapping layer_idx -> sae_weights
        tokens: Token IDs [batch_size]
        layers: List of layer indices to process
        all_features_dict: Dict mapping layer_idx -> set of all feature indices
    
    Returns:
        results_dict: Dict mapping layer_idx -> batch results dict
    """
    results_dict = {}
    
    for layer_idx in layers:
        site = f"resid_out_layer{layer_idx}"
        sae_weights = sae_weights_dict[layer_idx]
        all_features = all_features_dict[layer_idx]
        
        try:
            results = compare_entropies_for_batch(
                model, tokenizer, sae_weights, tokens, layer_idx,
                all_features, site
            )
            results_dict[layer_idx] = results
        except Exception as e:
            print(f"[WARN] Error processing layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            results_dict[layer_idx] = None
    
    return results_dict


def main(layers=None, num_batches=None, random_batches=True, random_seed=None):
    """
    Main function to compare entropies across multiple layers in a single pass.
    
    Args:
        layers: List of layer indices (e.g., [0, 1, 2, 3, 4, 5]) or None for all layers
        num_batches: Number of batches to process (if None, uses NUM_BATCHES)
        random_batches: If True, select batches randomly from the corpus
        random_seed: Random seed for reproducibility (if None, uses system default)
    """
    if layers is None:
        layers = [0, 1, 2, 3, 4, 5]
    
    if num_batches is None:
        num_batches = NUM_BATCHES
    
    # Generate unique ID based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = timestamp
    
    print(f"\n{'='*60}")
    print(f"[INFO] Comparing entropies for layers: {layers}")
    print(f"[INFO] Optimized multi-layer processing")
    print(f"{'='*60}")
    
    # 1. Setup - Load model and tokenizer ONCE
    print("[INFO] Loading Model and Tokenizer (once for all layers)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    d_model = model.config.hidden_size
    
    # 2. Load SAE weights for all layers ONCE
    print(f"[INFO] Loading SAE weights for {len(layers)} layers...")
    sae_weights_dict = load_all_sae_weights(layers, d_model)
    
    # Get total number of features for each layer
    all_features_dict = {}
    for layer_idx in layers:
        n_latent = sae_weights_dict[layer_idx]["dec_w"].shape[1]
        all_features_dict[layer_idx] = set(range(n_latent))
        print(f"[INFO] Layer {layer_idx}: {n_latent} features")
    
    # 3. Load data ONCE
    DATA_FILE = Path("wikitext-2-train.txt")
    if not DATA_FILE.exists():
        print("[ERROR] wikitext-2-train.txt not found.")
        sys.exit(1)
    
    print(f"[INFO] Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"[INFO] Tokenizing text...")
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_tokens = tokens.shape[0]
    print(f"[INFO] Total tokens available: {total_tokens}")
    
    # 4. Generate batch start indices
    print(f"\n[INFO] Processing {num_batches} batches of size {BATCH_SIZE}...")
    print(f"[INFO] Threshold: {THRESHOLD}")
    if random_batches:
        print(f"[INFO] Batch selection: Random")
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            print(f"[INFO] Random seed: {random_seed}")
    else:
        print(f"[INFO] Batch selection: Sequential (from beginning)")
    
    max_start = total_tokens - BATCH_SIZE
    if max_start <= 0:
        print("[ERROR] Not enough tokens for batch size")
        return
    
    if random_batches:
        available_starts = list(range(0, max_start + 1, BATCH_SIZE))
        if len(available_starts) < num_batches:
            available_starts = list(range(max_start + 1))
        batch_start_indices = random.sample(available_starts, min(num_batches, len(available_starts)))
        batch_start_indices.sort()
    else:
        batch_start_indices = [(batch_idx * (max_start // num_batches)) % max_start 
                              for batch_idx in range(num_batches)]
    
    # 5. Process batches - results organized by layer
    # Structure: batch_results_by_layer[layer_idx] = list of batch results
    batch_results_by_layer = {layer_idx: [] for layer_idx in layers}
    
    import time
    total_start_time = time.time()
    
    for batch_idx in range(num_batches):
        start_idx = batch_start_indices[batch_idx]
        chunk = tokens[start_idx : start_idx + BATCH_SIZE].to(DEVICE)
        
        print(f"\n[INFO] Processing batch {batch_idx + 1}/{num_batches} (start_idx={start_idx})...")
        batch_start_time = time.time()
        
        # Process this batch for all layers
        results_dict = process_batch_for_all_layers(
            model, tokenizer, sae_weights_dict, chunk, layers, all_features_dict
        )
        
        # Store results for each layer
        for layer_idx in layers:
            if results_dict[layer_idx] is not None:
                batch_results_by_layer[layer_idx].append({
                    "batch_idx": batch_idx,
                    "start_idx": start_idx,
                    **results_dict[layer_idx]
                })
        
        batch_elapsed = time.time() - batch_start_time
        print(f"  Batch processed in {batch_elapsed:.2f}s for {len(layers)} layers")
        
        # Print summary for this batch
        for layer_idx in layers:
            if results_dict[layer_idx] is not None:
                results = results_dict[layer_idx]
                print(f"  Layer {layer_idx}: {results['num_active_features']} active features, "
                      f"token entropy: {results['token_vector_entropy']:.4f} bits")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n[INFO] Total processing time: {total_elapsed:.2f}s")
    print(f"[INFO] Average time per batch: {total_elapsed/num_batches:.2f}s")
    print(f"[INFO] Average time per batch per layer: {total_elapsed/(num_batches*len(layers)):.2f}s")
    
    # 6. Create plots and save results for each layer separately
    print(f"\n[INFO] Creating plots and saving results for each layer...")
    
    all_output_files = {}
    
    for layer_idx in layers:
        site = f"resid_out_layer{layer_idx}"
        batch_results = batch_results_by_layer[layer_idx]
        
        if len(batch_results) == 0:
            print(f"[WARN] No batches processed successfully for layer {layer_idx}")
            continue
        
        # Create plots
        plots_dir = Path(f"entropy_plots_{site}_{unique_id}")
        plots_dir.mkdir(exist_ok=True)
        
        plot_paths = []
        for br in batch_results:
            batch_idx = br['batch_idx']
            plot_path = plot_batch_comparison(br, batch_idx, site, plots_dir)
            plot_paths.append(plot_path)
        
        print(f"[INFO] Layer {layer_idx}: Saved {len(plot_paths)} plots to {plots_dir}/")
        
        # Save batch index file
        batch_index_file = plots_dir / "batch_index.json"
        batch_index_data = {
            "site": site,
            "unique_id": unique_id,
            "timestamp": timestamp,
            "num_batches": len(batch_results),
            "batch_size": BATCH_SIZE,
            "random_batches": random_batches,
            "random_seed": random_seed,
            "batches": [
                {
                    "batch_idx": br['batch_idx'],
                    "start_idx": br['start_idx'],
                    "plot_file": f"batch_{br['batch_idx']:03d}.png",
                    "num_active_features": br['num_active_features'],
                    "token_vector_entropy": float(br['token_vector_entropy']),
                    "avg_feature_entropy": float(np.mean(list(br['feature_entropies'].values()))) if br['num_active_features'] > 0 else None
                }
                for br in batch_results
            ]
        }
        
        with open(batch_index_file, 'w') as f:
            json.dump(batch_index_data, f, indent=2)
        
        # Save results
        output_file = Path(f"entropy_comparison_{site}_{unique_id}.pt")
        output_data = {
            "batch_results": batch_results,
            "summary": {
                "site": site,
                "unique_id": unique_id,
                "timestamp": timestamp,
                "layer": layer_idx,
                "num_batches": len(batch_results),
                "mean_feature_entropy": float(np.mean([np.mean(list(br['feature_entropies'].values())) 
                                                      for br in batch_results if br['num_active_features'] > 0])) if any(br['num_active_features'] > 0 for br in batch_results) else None,
                "mean_token_vector_entropy": float(np.mean([br['token_vector_entropy'] for br in batch_results])),
            },
            "config": {
                "threshold": THRESHOLD,
                "batch_size": BATCH_SIZE,
                "total_features": len(all_features_dict[layer_idx]),
                "random_batches": random_batches,
                "random_seed": random_seed,
            },
            "plots_dir": str(plots_dir),
            "batch_start_indices": [br['start_idx'] for br in batch_results]
        }
        
        torch.save(output_data, output_file)
        all_output_files[layer_idx] = output_file
        print(f"[INFO] Layer {layer_idx}: Saved results to {output_file}")
    
    # 7. Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Processed {num_batches} batches for {len(layers)} layers")
    print(f"Total processing time: {total_elapsed:.2f}s")
    print(f"Output files:")
    for layer_idx in layers:
        if layer_idx in all_output_files:
            print(f"  Layer {layer_idx}: {all_output_files[layer_idx]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare entropies for multiple layers in a single pass")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                       help="Layer indices to process (e.g., 0 1 2 3 4 5). Default: all layers 0-5")
    parser.add_argument("--num-batches", type=int, default=None,
                       help="Number of batches to process (default: 10)")
    parser.add_argument("--sequential-batches", action="store_false", dest="random_batches",
                       help="Select batches sequentially from beginning instead of randomly")
    parser.add_argument("--random-seed", type=int, default=None,
                       help="Random seed for reproducible random batch selection")
    
    args = parser.parse_args()
    
    try:
        main(layers=args.layers, num_batches=args.num_batches,
             random_batches=args.random_batches, random_seed=args.random_seed)
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
