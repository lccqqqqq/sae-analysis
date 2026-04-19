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
NUM_BATCHES_TO_ANALYZE = 3  # Number of random batches to analyze
TOP_K_TOKENS = 10  # Number of top influential tokens to show per batch
SHOW_ACTUAL_TOKENS = True  # Set to True to process sample batches and show actual tokens

# Load the feature token influence data
print(f"Loading data for {SITE}...")
data_file = f"feature_token_influence_{SITE}.pt"
data = torch.load(data_file, map_location='cpu', weights_only=False)

feature_influences = data['feature_influences']
config = data.get('config', {})
BATCH_SIZE = config.get('batch_size', 64)

# Check if feature exists
if FEATURE_IDX not in feature_influences:
    available_features = sorted(feature_influences.keys())
    print(f"ERROR: Feature {FEATURE_IDX} not found in data.")
    print(f"Available features: {available_features[:20]}..." if len(available_features) > 20 else f"Available features: {available_features}")
    raise KeyError(f"Feature {FEATURE_IDX} not found")

# Get the feature data
feature_data = feature_influences[FEATURE_IDX]
all_influences = feature_data['all_influences']  # List of [seq_len] arrays, one per batch
num_samples = feature_data.get('num_samples', len(all_influences))

print(f"Feature {FEATURE_IDX}: {num_samples} batches available")

# Randomly select batches to analyze
if num_samples < NUM_BATCHES_TO_ANALYZE:
    print(f"Warning: Only {num_samples} batches available, analyzing all of them")
    batch_indices = list(range(num_samples))
else:
    batch_indices = random.sample(range(num_samples), NUM_BATCHES_TO_ANALYZE)
    batch_indices.sort()

print(f"Analyzing batches: {batch_indices}")

# Load tokenizer
print(f"\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

# If we want to show actual tokens, we need to process sample batches
# Since we don't have batch start positions, we'll process random samples from the text
if SHOW_ACTUAL_TOKENS:
    print("Loading model to process sample batches for token decoding...")
    DATA_FILE = Path("wikitext-2-train.txt")
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        all_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        print(f"Loaded {len(all_tokens)} tokens")
        
        # Process a few sample batches to show tokens
        # We'll use random positions from the text
        max_start = len(all_tokens) - BATCH_SIZE
        sample_starts = random.sample(range(max_start), min(3, max_start // BATCH_SIZE))
        sample_batches = {}
        for start_idx in sample_starts:
            batch_tokens = all_tokens[start_idx:start_idx + BATCH_SIZE]
            sample_batches[start_idx] = batch_tokens
        print(f"Prepared {len(sample_batches)} sample batches for token display")
    else:
        print(f"Warning: {DATA_FILE} not found. Cannot show actual tokens.")
        SHOW_ACTUAL_TOKENS = False
        sample_batches = {}
else:
    sample_batches = {}

# Analyze each selected batch
print(f"\n{'='*80}")
print(f"Analysis of Feature {FEATURE_IDX} - Tokens with Strong Influence")
print(f"{'='*80}")
print(f"\nNote: Influence shows how each token position affects the feature activation.")
print(f"Higher influence = stronger effect on the feature at the last token position.\n")

for batch_idx in batch_indices:
    influence_dist = np.array(all_influences[batch_idx])  # [seq_len]
    seq_len = len(influence_dist)
    
    print(f"\n{'='*80}")
    print(f"Batch {batch_idx} (Sequence length: {seq_len})")
    print(f"{'='*80}")
    
    # Get top K token positions by influence
    top_k_indices = np.argsort(influence_dist)[-TOP_K_TOKENS:][::-1]
    
    print(f"\nTop {len(top_k_indices)} token positions by influence:")
    print(f"{'Position':<10} | {'Influence':<12} | {'Normalized':<12} | {'Token Text'}")
    print("-" * 90)
    
    # Normalize influence for better interpretation
    total_influence = np.sum(influence_dist)
    
    for pos_idx in top_k_indices:
        influence_val = influence_dist[pos_idx]
        normalized = influence_val / total_influence if total_influence > 0 else 0
        
        # Try to show token text from sample batches
        if SHOW_ACTUAL_TOKENS and sample_batches:
            # Use first sample batch as example (since we don't know exact batch start)
            sample_tokens = list(sample_batches.values())[0]
            if pos_idx < len(sample_tokens):
                token_id = sample_tokens[pos_idx].item()
                token_text = tokenizer.decode([token_id])
                # Clean up token text for display
                token_text = repr(token_text[:50])  # Limit length
            else:
                token_text = f"Position {pos_idx} (out of range)"
        else:
            token_text = f"Position {pos_idx}/{seq_len-1}"
        
        print(f"{pos_idx:<10} | {influence_val:<12.6f} | {normalized:<12.4%} | {token_text}")
    
    # Show influence distribution statistics
    print(f"\nInfluence distribution statistics:")
    print(f"  Total influence: {total_influence:.6f}")
    print(f"  Mean: {influence_dist.mean():.6f}")
    print(f"  Max: {influence_dist.max():.6f} (at position {np.argmax(influence_dist)})")
    print(f"  Min: {influence_dist.min():.6f}")
    print(f"  Std: {influence_dist.std():.6f}")
    
    # Show which positions have the strongest influence (top 3)
    top3_positions = np.argsort(influence_dist)[-3:][::-1]
    print(f"\nTop 3 positions: {top3_positions.tolist()}")
    print(f"  These positions contribute {np.sum(influence_dist[top3_positions])/total_influence*100:.1f}% of total influence")
    
    # Visualize influence distribution
    plt.figure(figsize=(12, 4))
    plt.plot(influence_dist, 'b-', linewidth=1.5, alpha=0.7, label='Influence')
    plt.scatter(top_k_indices, influence_dist[top_k_indices], 
               color='red', s=50, zorder=5, label=f'Top {TOP_K_TOKENS} positions')
    plt.xlabel('Token Position in Batch', fontsize=11, fontweight='bold')
    plt.ylabel('Influence Value', fontsize=11, fontweight='bold')
    plt.title(f'Feature {FEATURE_IDX} - Batch {batch_idx}: Influence Distribution', 
              fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

print(f"\n{'='*80}")
print("Summary:")
print(f"- Analyzed {len(batch_indices)} batches for Feature {FEATURE_IDX}")
print(f"- Influence values show how each token position affects the feature")
print(f"- Higher influence = stronger contribution to feature activation")
print(f"- Positions are relative to the batch (0 = first token, {seq_len-1} = last token)")
if not SHOW_ACTUAL_TOKENS or not sample_batches:
    print(f"- Note: Token text shown is from sample batches, not exact batch matches")
print(f"{'='*80}")
