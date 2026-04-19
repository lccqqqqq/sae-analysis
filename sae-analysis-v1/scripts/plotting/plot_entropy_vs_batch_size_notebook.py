# Jupyter Notebook Code Block for Plotting Entropy vs Batch Size
# Copy this entire block into a Jupyter notebook cell

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# Load data from the saved file
data_file = "entropy_vs_batch_size_resid_out_layer3_20260111_155223.pt"
data = torch.load(data_file, map_location="cpu")

# Extract data
results_by_batch_size = data["results_by_batch_size"]
site = data["summary"]["site"]

# Collect all features that appear at any batch size
all_feature_indices = set()
for result in results_by_batch_size.values():
    all_feature_indices.update(result['feature_entropies'].keys())

all_feature_indices = sorted(all_feature_indices)
batch_sizes = sorted(results_by_batch_size.keys())

# Create color map for features (consistent across batch sizes)
n_features = len(all_feature_indices)
if n_features > 0:
    # Use tab20 colormap for up to 20 features, then cycle
    if n_features <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_features))
    else:
        # For more features, use a larger colormap or cycle
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        # Extend by cycling
        while len(colors) < n_features:
            colors = np.vstack([colors, plt.cm.tab20(np.linspace(0, 1, 20))])
        colors = colors[:n_features]
    
    feature_color_map = {feat_idx: colors[i] for i, feat_idx in enumerate(all_feature_indices)}
else:
    feature_color_map = {}

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot entropy vs batch size for each feature
for feat_idx in all_feature_indices:
    entropies = []
    batch_sizes_with_feature = []
    
    for batch_size in batch_sizes:
        result = results_by_batch_size[batch_size]
        if feat_idx in result['feature_entropies']:
            entropies.append(result['feature_entropies'][feat_idx])
            batch_sizes_with_feature.append(batch_size)
    
    if len(entropies) > 0:
        color = feature_color_map[feat_idx]
        ax.plot(batch_sizes_with_feature, entropies, 'o-',
                color=color, label=f'Feature {feat_idx}',
                linewidth=2, markersize=6, alpha=0.7)

# Add maximal entropy line (log2(batch_size) for uniform distribution)
max_entropy_batch_sizes = np.array(batch_sizes)
max_entropies = np.log2(max_entropy_batch_sizes)
ax.plot(max_entropy_batch_sizes, max_entropies, 'k--', 
        linewidth=2, label='Maximal Entropy (log₂(n))', alpha=0.8)

ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Entropy (bits)', fontsize=12)
ax.set_title(f'{site} - Feature Entropy vs Batch Size',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add legend (limit to top features by total activation across all batch sizes)
if len(all_feature_indices) > 0:
    # Compute total activation for each feature across all batch sizes
    feature_total_activations = {}
    for feat_idx in all_feature_indices:
        total_act = 0.0
        for batch_size in batch_sizes:
            result = results_by_batch_size[batch_size]
            if feat_idx in result['feature_activations']:
                total_act += result['feature_activations'][feat_idx]
        feature_total_activations[feat_idx] = total_act
    
    # Get top 10 features by total activation
    top_features = sorted(feature_total_activations.items(),
                         key=lambda x: x[1], reverse=True)[:10]
    top_feature_indices = {idx for idx, _ in top_features}
    
    # Create legend with only top features
    legend_elements = []
    for feat_idx in all_feature_indices:
        if feat_idx in top_feature_indices:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=feature_color_map[feat_idx],
                          markersize=8, markeredgecolor='black',
                          markeredgewidth=0.5, linewidth=2,
                          label=f'Feature {feat_idx}')
            )
    
    # Add maximal entropy to legend
    legend_elements.append(
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2,
                  label='Maximal Entropy (log₂(n))')
    )
    
    if len(legend_elements) > 0:
        ax.legend(handles=legend_elements, fontsize=9, loc='best', ncol=2)

plt.tight_layout()
plt.show()
