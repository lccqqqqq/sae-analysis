"""
Jupyter notebook code block to plot entropy of leading features vs depth (layer 0-5).

This code:
1. Loads entropy comparison data from all 6 layers
2. Identifies leading features (top features by activation)
3. Plots entropy vs depth with consistent colors per feature
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import glob

# Configuration
LAYERS = [0, 1, 2, 3, 4, 5]
NUM_LEADING_FEATURES = 10  # Number of top features to track

# Find the most recent entropy comparison files for each layer
def find_latest_entropy_file(layer):
    """Find the most recent entropy comparison file for a given layer."""
    pattern = f"entropy_comparison_resid_out_layer{layer}_*.pt"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No entropy comparison file found for layer {layer}")
    # Sort by modification time and get the most recent
    latest = max(files, key=lambda f: Path(f).stat().st_mtime)
    return latest

# Load data from all layers
print("Loading entropy comparison data from all layers...")
layer_data = {}
for layer in LAYERS:
    file_path = find_latest_entropy_file(layer)
    print(f"  Layer {layer}: {file_path}")
    data = torch.load(file_path, map_location='cpu')
    layer_data[layer] = data

# Extract feature entropies and activations across all layers
# We'll identify leading features based on their average activation across all layers
feature_activation_sums = defaultdict(float)
feature_entropy_data = defaultdict(dict)  # layer -> feature_idx -> entropy
feature_activation_data = defaultdict(dict)  # layer -> feature_idx -> activation

for layer in LAYERS:
    batch_results = layer_data[layer]['batch_results']
    
    # Aggregate across all batches for this layer
    for batch_result in batch_results:
        feature_entropies = batch_result.get('feature_entropies', {})
        feature_activations = batch_result.get('feature_activations', {})
        
        for feat_idx in feature_entropies.keys():
            # Store entropy and activation for this feature in this layer
            if feat_idx not in feature_entropy_data[layer]:
                feature_entropy_data[layer][feat_idx] = []
                feature_activation_data[layer][feat_idx] = []
            
            feature_entropy_data[layer][feat_idx].append(feature_entropies[feat_idx])
            if feat_idx in feature_activations:
                feature_activation_data[layer][feat_idx].append(feature_activations[feat_idx])
                # Accumulate activation for leading feature selection
                feature_activation_sums[feat_idx] += feature_activations[feat_idx]

# Average entropies and activations per feature per layer
feature_entropy_avg = defaultdict(dict)
feature_activation_avg = defaultdict(dict)

for layer in LAYERS:
    for feat_idx in feature_entropy_data[layer]:
        if feature_entropy_data[layer][feat_idx]:
            feature_entropy_avg[layer][feat_idx] = np.mean(feature_entropy_data[layer][feat_idx])
        if feature_activation_data[layer][feat_idx]:
            feature_activation_avg[layer][feat_idx] = np.mean(feature_activation_data[layer][feat_idx])

# Identify leading features (top features by total activation across all layers)
# Or use features that appear in multiple layers
feature_appearance_count = defaultdict(int)
for layer in LAYERS:
    for feat_idx in feature_entropy_avg[layer]:
        feature_appearance_count[feat_idx] += 1

# Get top features by average activation (weighted by appearance)
feature_scores = {}
for feat_idx in feature_activation_sums:
    if feature_appearance_count[feat_idx] >= 2:  # Must appear in at least 2 layers
        feature_scores[feat_idx] = feature_activation_sums[feat_idx] / feature_appearance_count[feat_idx]

# Sort and get top N features
sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
leading_features = [feat_idx for feat_idx, _ in sorted_features[:NUM_LEADING_FEATURES]]

print(f"\nIdentified {len(leading_features)} leading features: {leading_features}")

# Prepare data for plotting
plot_data = {}  # feature_idx -> (layers, entropies)
for feat_idx in leading_features:
    layers_list = []
    entropies_list = []
    
    for layer in LAYERS:
        if feat_idx in feature_entropy_avg[layer]:
            layers_list.append(layer)
            entropies_list.append(feature_entropy_avg[layer][feat_idx])
    
    if layers_list:  # Only include if feature appears in at least one layer
        plot_data[feat_idx] = (layers_list, entropies_list)

# Create the plot
plt.figure(figsize=(12, 8))

# Assign consistent colors to features
if len(leading_features) > 0:
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(leading_features)] if len(leading_features) <= 10 else plt.cm.tab20(np.linspace(0, 1, 20))[:len(leading_features)]
    feature_colors = {feat_idx: colors[i] for i, feat_idx in enumerate(leading_features)}
else:
    feature_colors = {}

# Plot each leading feature
for feat_idx in leading_features:
    if feat_idx in plot_data:
        layers_list, entropies_list = plot_data[feat_idx]
        color = feature_colors[feat_idx]
        plt.plot(layers_list, entropies_list, 'o-', color=color, linewidth=2, 
                markersize=8, label=f'Feature {feat_idx}', alpha=0.8)

plt.xlabel('Layer Depth', fontsize=12, fontweight='bold')
plt.ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
plt.title('Entropy of Leading Features vs Layer Depth', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.xticks(LAYERS, [f'Layer {l}' for l in LAYERS])
plt.tight_layout()

plt.show()

print(f"\nPlot created with {len(plot_data)} leading features tracked across layers.")
