# Jupyter Notebook Code Block: Plot Entropy of Leading Features vs Depth
# Copy this entire block into a Jupyter notebook cell

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
    latest = max(files, key=lambda f: Path(f).stat().st_mtime)
    return latest

# Load data from all layers
print("Loading entropy comparison data from all layers...")
layer_data = {}
for layer in LAYERS:
    file_path = find_latest_entropy_file(layer)
    print(f"  Layer {layer}: {Path(file_path).name}")
    data = torch.load(file_path, map_location='cpu')
    layer_data[layer] = data

# Extract feature entropies and activations across all layers
feature_activation_sums = defaultdict(float)
feature_entropy_data = defaultdict(lambda: defaultdict(list))
feature_activation_data = defaultdict(lambda: defaultdict(list))

for layer in LAYERS:
    batch_results = layer_data[layer]['batch_results']
    for batch_result in batch_results:
        feature_entropies = batch_result.get('feature_entropies', {})
        feature_activations = batch_result.get('feature_activations', {})
        
        for feat_idx in feature_entropies.keys():
            feature_entropy_data[layer][feat_idx].append(feature_entropies[feat_idx])
            if feat_idx in feature_activations:
                feature_activation_data[layer][feat_idx].append(feature_activations[feat_idx])
                feature_activation_sums[feat_idx] += feature_activations[feat_idx]

# Average entropies per feature per layer
feature_entropy_avg = defaultdict(dict)
for layer in LAYERS:
    for feat_idx in feature_entropy_data[layer]:
        if feature_entropy_data[layer][feat_idx]:
            feature_entropy_avg[layer][feat_idx] = np.mean(feature_entropy_data[layer][feat_idx])

# Identify leading features (top features by total activation, must appear in at least 2 layers)
feature_appearance_count = defaultdict(int)
for layer in LAYERS:
    for feat_idx in feature_entropy_avg[layer]:
        feature_appearance_count[feat_idx] += 1

feature_scores = {}
for feat_idx in feature_activation_sums:
    if feature_appearance_count[feat_idx] >= 2:
        feature_scores[feat_idx] = feature_activation_sums[feat_idx] / feature_appearance_count[feat_idx]

sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
leading_features = [feat_idx for feat_idx, _ in sorted_features[:NUM_LEADING_FEATURES]]

print(f"\nIdentified {len(leading_features)} leading features: {leading_features}")

# Prepare data for plotting
plot_data = {}
for feat_idx in leading_features:
    layers_list = []
    entropies_list = []
    for layer in LAYERS:
        if feat_idx in feature_entropy_avg[layer]:
            layers_list.append(layer)
            entropies_list.append(feature_entropy_avg[layer][feat_idx])
    if layers_list:
        plot_data[feat_idx] = (layers_list, entropies_list)

# Create the plot
plt.figure(figsize=(12, 8))

# Assign consistent colors to features
if len(leading_features) > 0:
    n_colors = len(leading_features)
    if n_colors <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_colors]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_colors]
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
