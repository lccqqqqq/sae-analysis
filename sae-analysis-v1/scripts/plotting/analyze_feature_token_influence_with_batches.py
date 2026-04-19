# Analyze which tokens have strong influence on a given feature
# Uses entropy comparison file as input, picks a leading feature, then shows tokens
# Copy this into a Jupyter notebook cell

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from pathlib import Path
import random
from collections import defaultdict

# Configuration - change these as needed
ENTROPY_FILE = "entropy_comparison_resid_out_layer3_20260110_194042.pt"  # Input file from compare_entropies.py
FEATURE_IDX = None  # Set to None to auto-pick top feature, or specify a feature index
NUM_BATCHES_TO_ANALYZE = 3  # Number of random batches to analyze
TOP_K_TOKENS = 10  # Number of top influential tokens to show per batch

# Load entropy comparison data
print(f"Loading entropy comparison data from {ENTROPY_FILE}...")
entropy_data = torch.load(ENTROPY_FILE, map_location='cpu', weights_only=False)
batch_results = entropy_data.get('batch_results', [])
batch_start_indices = entropy_data.get('batch_start_indices', [])
summary = entropy_data.get('summary', {})
config = entropy_data.get('config', {})

site = summary.get('site', config.get('site', 'unknown'))
layer_idx = summary.get('layer', config.get('layer', None))
BATCH_SIZE = config.get('batch_size', 64)
THRESHOLD = config.get('threshold', 0.2)

print(f"Site: {site}, Layer: {layer_idx}")
print(f"Found {len(batch_results)} batches in entropy comparison data")

# Create mapping from batch index to start_idx
batch_to_start_idx = {}
if batch_start_indices:
    # Use the direct list if available
    for batch_idx, start_idx in enumerate(batch_start_indices):
        batch_to_start_idx[batch_idx] = start_idx
else:
    # Fall back to extracting from batch_results
    for br in batch_results:
        batch_idx = br.get('batch_idx', None)
        start_idx = br.get('start_idx', None)
        if batch_idx is not None and start_idx is not None:
            batch_to_start_idx[batch_idx] = start_idx

print(f"Found {len(batch_to_start_idx)} batches with start positions")

# Extract leading features from entropy comparison data
print(f"\nExtracting leading features from entropy comparison data...")
feature_activations = defaultdict(list)  # feature_idx -> [activations across batches]
feature_entropies = defaultdict(list)  # feature_idx -> [entropies across batches]

for br in batch_results:
    feat_ents = br.get('feature_entropies', {})
    feat_acts = br.get('feature_activations', {})
    
    for feat_idx in feat_ents.keys():
        feature_entropies[feat_idx].append(feat_ents[feat_idx])
        if feat_idx in feat_acts:
            feature_activations[feat_idx].append(feat_acts[feat_idx])

# Compute average activation and entropy for each feature
feature_stats = {}
for feat_idx in feature_activations.keys():
    avg_activation = np.mean(feature_activations[feat_idx])
    avg_entropy = np.mean(feature_entropies[feat_idx])
    num_batches = len(feature_activations[feat_idx])
    feature_stats[feat_idx] = {
        'avg_activation': avg_activation,
        'avg_entropy': avg_entropy,
        'num_batches': num_batches
    }

# Sort features by average activation (leading features)
sorted_features = sorted(feature_stats.items(), 
                        key=lambda x: x[1]['avg_activation'], 
                        reverse=True)

print(f"\nTop 10 leading features (by average activation):")
print(f"{'Feature':<10} | {'Avg Activation':<15} | {'Avg Entropy':<15} | {'Num Batches':<12}")
print("-" * 70)
for feat_idx, stats in sorted_features[:10]:
    print(f"{feat_idx:<10} | {stats['avg_activation']:<15.6f} | {stats['avg_entropy']:<15.4f} | {stats['num_batches']:<12}")

# Select feature to analyze
if FEATURE_IDX is None:
    # Auto-pick the top feature by activation
    FEATURE_IDX = sorted_features[0][0]
    print(f"\nAuto-selected Feature {FEATURE_IDX} (top by activation)")
else:
    if FEATURE_IDX not in feature_stats:
        available = sorted(list(feature_stats.keys()))
        print(f"\nERROR: Feature {FEATURE_IDX} not found in entropy comparison data.")
        print(f"Available features: {available[:20]}..." if len(available) > 20 else f"Available features: {available}")
        raise KeyError(f"Feature {FEATURE_IDX} not found")
    print(f"\nUsing Feature {FEATURE_IDX}")

# Load the feature token influence data
print(f"\nLoading feature influence data for {site}...")
influence_file = f"feature_token_influence_{site}.pt"
influence_data = torch.load(influence_file, map_location='cpu', weights_only=False)

feature_influences = influence_data.get('feature_influences', {})
influence_config = influence_data.get('config', {})

# Check if feature exists in influence data
if FEATURE_IDX not in feature_influences:
    available_features = sorted(feature_influences.keys())
    print(f"ERROR: Feature {FEATURE_IDX} not found in influence data.")
    print(f"Available features: {available_features[:20]}..." if len(available_features) > 20 else f"Available features: {available_features}")
    raise KeyError(f"Feature {FEATURE_IDX} not found in influence data")

# Get the feature data
feature_data = feature_influences[FEATURE_IDX]
all_influences = feature_data['all_influences']  # List of [seq_len] arrays, one per batch
num_samples = feature_data.get('num_samples', len(all_influences))

print(f"Feature {FEATURE_IDX}: {num_samples} batches available in influence data")

# Randomly select batches to analyze from stored data
if num_samples < NUM_BATCHES_TO_ANALYZE:
    print(f"Warning: Only {num_samples} batches available, analyzing all of them")
    batch_indices = list(range(num_samples))
else:
    batch_indices = random.sample(range(num_samples), NUM_BATCHES_TO_ANALYZE)
    batch_indices.sort()

print(f"Analyzing stored batches: {batch_indices}")

# Load tokenizer and text data to decode actual tokens
print(f"\nLoading tokenizer and text data...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

DATA_FILE = Path("wikitext-2-train.txt")
if not DATA_FILE.exists():
    raise FileNotFoundError(f"{DATA_FILE} not found")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()
all_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
print(f"Loaded {len(all_tokens)} tokens")

# Analyze each selected batch using stored influence data
print(f"\n{'='*80}")
print(f"Analysis of Feature {FEATURE_IDX} - Tokens with Strong Influence")
print(f"{'='*80}")
print(f"\nUsing stored influence data + batch positions from {ENTROPY_FILE}")
print(f"Note: Influence shows how each token position affects the feature activation.")
print(f"Higher influence = stronger effect on the feature at the last token position.\n")

for batch_idx in batch_indices:
    # Use stored influence data directly
    influence_dist = np.array(all_influences[batch_idx])  # [seq_len]
    seq_len = len(influence_dist)
    
    # Get start position for this batch (if available)
    # Note: batch_idx in influence data might not match batch_idx in entropy data
    # We'll try to match by index, but it may not be perfect
    start_idx = batch_to_start_idx.get(batch_idx, None)
    
    print(f"\n{'='*80}")
    if start_idx is not None:
        print(f"Batch {batch_idx} (start_idx={start_idx}, sequence length: {seq_len})")
    else:
        print(f"Batch {batch_idx} (start_idx unknown, sequence length: {seq_len})")
    print(f"{'='*80}")
    
    # Get top K token positions by influence
    top_k_indices = np.argsort(influence_dist)[-TOP_K_TOKENS:][::-1]
    
    print(f"\nTop {len(top_k_indices)} token positions by influence:")
    print(f"{'Position':<10} | {'Influence':<12} | {'Normalized':<12} | {'Token Text'}")
    print("-" * 90)
    
    # Normalize influence for better interpretation
    total_influence = np.sum(influence_dist)
    
    # Get actual tokens if we have start_idx
    if start_idx is not None and start_idx + seq_len <= len(all_tokens):
        batch_tokens = all_tokens[start_idx:start_idx + seq_len]
        has_tokens = True
    else:
        has_tokens = False
        batch_tokens = None
    
    for pos_idx in top_k_indices:
        influence_val = influence_dist[pos_idx]
        normalized = influence_val / total_influence if total_influence > 0 else 0
        
        # Decode actual token if available
        if has_tokens and batch_tokens is not None:
            token_id = int(batch_tokens[pos_idx].item())
            token_text = tokenizer.decode([token_id])
            # Clean up for display
            token_text = repr(token_text.replace('\n', '\\n').replace('\r', '\\r')[:40])
            token_display = token_text
        else:
            token_display = f"Position {pos_idx}/{seq_len-1}"
        
        print(f"{pos_idx:<10} | {influence_val:<12.6f} | {normalized:<12.4%} | {token_display}")
    
    # Show context around top tokens if we have actual tokens
    if has_tokens and batch_tokens is not None:
        print(f"\nContext around top influential tokens:")
        for pos_idx in top_k_indices[:5]:  # Show top 5
            context_start = max(0, pos_idx - 3)
            context_end = min(seq_len, pos_idx + 4)
            context_tokens = batch_tokens[context_start:context_end]
            context_text = tokenizer.decode(context_tokens.tolist())
            # Mark the influential token position
            context_parts = context_text.split(' ')
            marker_pos = pos_idx - context_start
            if marker_pos < len(context_parts):
                context_parts[marker_pos] = f"**{context_parts[marker_pos]}**"
                context_text = ' '.join(context_parts)
            print(f"  Position {pos_idx}: ...{context_text[:100]}...")
    
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
    plt.figure(figsize=(12, 5))
    plt.plot(influence_dist, 'b-', linewidth=1.5, alpha=0.7, label='Influence')
    plt.scatter(top_k_indices, influence_dist[top_k_indices], 
               color='red', s=50, zorder=5, label=f'Top {TOP_K_TOKENS} positions')
    plt.xlabel('Token Position in Batch', fontsize=11, fontweight='bold')
    plt.ylabel('Influence Value', fontsize=11, fontweight='bold')
    title = f'Feature {FEATURE_IDX} - Batch {batch_idx}'
    if start_idx is not None:
        title += f' (start_idx={start_idx})'
    title += '\nInfluence Distribution'
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

print(f"\n{'='*80}")
print("Summary:")
print(f"- Analyzed Feature {FEATURE_IDX} (selected from {ENTROPY_FILE})")
print(f"- Analyzed {len(batch_indices)} batches")
print(f"- Used stored influence data (no recomputation needed)")
print(f"- Used batch positions from entropy comparison file to decode actual tokens")
print(f"- Influence values show how each token position affects the feature")
print(f"- Higher influence = stronger contribution to feature activation")
print(f"- Positions are relative to the batch (0 = first token, {seq_len-1} = last token)")
print(f"{'='*80}")
