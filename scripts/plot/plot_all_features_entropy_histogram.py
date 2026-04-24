# Plot histogram of entropy distribution for all available features (or top 10)
# Copy this into a Jupyter notebook cell

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Input parameters - change these as needed
SITE = "resid_out_layer3"  # Change this to desired site/layer
MAX_FEATURES_TO_PLOT = 10  # Maximum number of features to plot

# Load the data
print(f"Loading data for {SITE}...")
data_file = f"feature_token_influence_{SITE}.pt"
data = torch.load(data_file, map_location='cpu', weights_only=False)

feature_influences = data['feature_influences']
config = data.get('config', {})

# Get all available features
available_features = sorted(feature_influences.keys())
print(f"Total available features: {len(available_features)}")

# Sort features by number of batches (num_samples) - features activated in most batches
feature_batch_counts = []
for feat_idx in available_features:
    feature_data = feature_influences[feat_idx]
    num_samples = feature_data.get('num_samples', 0)
    if 'all_influences' in feature_data:
        feature_batch_counts.append((feat_idx, num_samples))

# Sort by num_samples (descending)
feature_batch_counts.sort(key=lambda x: x[1], reverse=True)

# Select features to plot
if len(feature_batch_counts) <= MAX_FEATURES_TO_PLOT:
    features_to_plot = [feat_idx for feat_idx, _ in feature_batch_counts]
    print(f"Plotting all {len(features_to_plot)} features")
else:
    features_to_plot = [feat_idx for feat_idx, _ in feature_batch_counts[:MAX_FEATURES_TO_PLOT]]
    print(f"Plotting top {MAX_FEATURES_TO_PLOT} features (by number of batches)")
    print(f"Top features: {features_to_plot}")

# Compute entropy for each batch
def compute_feature_entropy(influence_distribution):
    """
    Compute entropy of a feature's influence distribution.
    
    Args:
        influence_distribution: [seq_len] array of J(t,z|t') values
    
    Returns:
        entropy: Shannon entropy in bits (base 2)
    """
    eps = 1e-12
    J_sum = np.sum(influence_distribution) + eps
    P = (influence_distribution + eps) / J_sum
    
    # Compute entropy: H = -∑ P log P
    entropy = stats.entropy(P, base=2)
    return entropy

# Compute entropies for all selected features
print("\nComputing entropies for each feature...")
all_feature_entropies = {}

for feat_idx in features_to_plot:
    feature_data = feature_influences[feat_idx]
    all_influences = feature_data['all_influences']
    num_samples = feature_data.get('num_samples', len(all_influences))
    
    entropies = []
    for influence_dist in all_influences:
        entropy = compute_feature_entropy(np.array(influence_dist))
        entropies.append(entropy)
    
    all_feature_entropies[feat_idx] = np.array(entropies)
    print(f"  Feature {feat_idx}: {len(entropies)} batches, mean entropy = {np.mean(entropies):.4f} bits")

# Create subplots for histograms
n_features = len(features_to_plot)
n_cols = min(3, n_features)
n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
if n_features == 1:
    axes = [axes]
elif n_rows == 1:
    axes = axes if isinstance(axes, np.ndarray) else [axes]
else:
    axes = axes.flatten()

for idx, feat_idx in enumerate(features_to_plot):
    ax = axes[idx]
    entropies = all_feature_entropies[feat_idx]
    
    # Create histogram
    n, bins, patches = ax.hist(entropies, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    
    # Add vertical dashed line at mean entropy
    mean_entropy = entropies.mean()
    ax.axvline(mean_entropy, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_entropy:.3f} bits', alpha=0.8)
    
    ax.set_xlabel('Entropy (bits)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Feature {feat_idx}\n(Mean: {mean_entropy:.3f} bits, N={len(entropies)})', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=8)

# Hide unused subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Entropy Distributions for Top Features ({SITE})', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print(f"\nPlot complete for {len(features_to_plot)} features in {SITE}")
