# Plot histogram of entropy distribution for a given feature across batches
# Copy this into a Jupyter notebook cell

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Input parameters - change these as needed
SITE = "resid_out_layer3"  # Change this to desired site/layer
FEATURE_IDX = 100  # Change this to desired feature index

# Load the data
print(f"Loading data for {SITE}...")
data_file = f"feature_token_influence_{SITE}.pt"
data = torch.load(data_file, map_location='cpu', weights_only=False)

feature_influences = data['feature_influences']
config = data.get('config', {})

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

print(f"Feature {FEATURE_IDX}: {num_samples} batches")

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

# Compute entropies for all batches
print("Computing entropies for each batch...")
entropies = []
for influence_dist in all_influences:
    entropy = compute_feature_entropy(np.array(influence_dist))
    entropies.append(entropy)

entropies = np.array(entropies)
print(f"Computed {len(entropies)} entropies")
print(f"Entropy range: {entropies.min():.4f} to {entropies.max():.4f} bits")
print(f"Mean entropy: {entropies.mean():.4f} ± {entropies.std():.4f} bits")

# Create histogram
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(entropies, bins=50, edgecolor='black', alpha=0.7, color='skyblue')

plt.xlabel('Entropy (bits)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title(f'Entropy Distribution for Feature {FEATURE_IDX} ({SITE})', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add statistics text
stats_text = f'Mean: {entropies.mean():.3f} bits\nStd: {entropies.std():.3f} bits\nN: {len(entropies)}'
plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=10, family='monospace')

plt.tight_layout()
plt.show()

print(f"\nPlot complete for Feature {FEATURE_IDX} in {SITE}")
