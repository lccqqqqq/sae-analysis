# Analyze which tokens have strong influence on a given feature
# Copy this into a Jupyter notebook cell

import torch
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
import random

# Configuration - change these as needed
SITE = "resid_out_layer3"  # Change this to desired site/layer
FEATURE_IDX = 100  # Change this to desired feature index
NUM_BATCHES_TO_ANALYZE = 3  # Number of random batches to analyze
TOP_K_TOKENS = 10  # Number of top influential tokens to show per batch
INFLUENCE_THRESHOLD = None  # Show tokens above this threshold (None = use top K)

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

# Load tokenizer and text data to get actual tokens
print(f"\nLoading tokenizer and text data...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

DATA_FILE = Path("wikitext-2-train.txt")
if not DATA_FILE.exists():
    print(f"Warning: {DATA_FILE} not found. Cannot decode actual token text.")
    print("Showing influence values and positions only.")
    tokens_available = False
    tokens = None
else:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    tokens_available = True
    print(f"Loaded {len(tokens)} tokens from {DATA_FILE}")
    print(f"Note: To decode exact tokens, we need batch start positions (not stored in data file)")
    print(f"Showing relative positions and influence patterns instead.")

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
    if INFLUENCE_THRESHOLD is not None:
        # Show all tokens above threshold
        top_k_indices = np.where(influence_dist >= INFLUENCE_THRESHOLD)[0]
        top_k_indices = top_k_indices[np.argsort(influence_dist[top_k_indices])[::-1]]
        if len(top_k_indices) > TOP_K_TOKENS:
            top_k_indices = top_k_indices[:TOP_K_TOKENS]
    else:
        # Get top K by value
        top_k_indices = np.argsort(influence_dist)[-TOP_K_TOKENS:][::-1]
    
    print(f"\nTop {len(top_k_indices)} token positions by influence:")
    print(f"{'Position':<10} | {'Influence':<12} | {'Normalized':<12} | {'Token Info'}")
    print("-" * 70)
    
    # Normalize influence for better interpretation
    total_influence = np.sum(influence_dist)
    
    for pos_idx in top_k_indices:
        influence_val = influence_dist[pos_idx]
        normalized = influence_val / total_influence if total_influence > 0 else 0
        
        # Try to show token info if we have tokens (approximate)
        if tokens_available and tokens is not None:
            # We don't know the exact batch start, but we can show the pattern
            # The influence is relative to positions in the batch
            token_info = f"Position {pos_idx}/{seq_len-1}"
        else:
            token_info = f"Position {pos_idx}/{seq_len-1}"
        
        print(f"{pos_idx:<10} | {influence_val:<12.6f} | {normalized:<12.4%} | {token_info}")
    
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

print(f"\n{'='*80}")
print("Summary:")
print(f"- Analyzed {len(batch_indices)} batches for Feature {FEATURE_IDX}")
print(f"- Influence values show how each token position affects the feature")
print(f"- Higher influence = stronger contribution to feature activation")
print(f"- Positions are relative to the batch (0 = first token, {BATCH_SIZE-1} = last token)")
print(f"{'='*80}")
