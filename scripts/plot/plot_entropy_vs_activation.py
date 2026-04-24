# Plot averaged entropy vs activation count
# Copy this into a Jupyter notebook cell

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Input parameter - change this as needed
SITE = "resid_out_layer3"  # Change this to desired site/layer

# 1. Load feature token influence data (for entropy)
print(f"Loading entropy data for {SITE}...")
influence_data = torch.load(f'feature_token_influence_{SITE}.pt', map_location='cpu', weights_only=False)
feature_influences = influence_data['feature_influences']

# 2. Load feature sparsity data (for activation probabilities)
print(f"Loading activation data for {SITE}...")
sparsity_data = torch.load(f'feature_sparsity_data_{SITE}.pt', map_location='cpu', weights_only=False)
feature_counts = sparsity_data['feature_counts'].cpu().numpy()  # Number of activations per feature
total_tokens = sparsity_data.get('total_tokens', len(feature_counts))

print(f"Total features in sparsity data: {len(feature_counts)}")
print(f"Total tokens: {total_tokens}")

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

# 3. Compute average entropy for each feature
print("Computing average entropy for each feature...")
avg_entropies = []
activation_probabilities = []
feature_indices = []

for feat_idx, info in feature_influences.items():
    if 'all_influences' not in info:
        continue
    
    # Get all influence traces
    all_influences = info['all_influences']  # List of [seq_len] arrays
    
    # Compute entropy for each batch
    entropies = []
    for influence_dist in all_influences:
        entropy = compute_feature_entropy(np.array(influence_dist))
        entropies.append(entropy)
    
    # Average entropy
    avg_entropy = np.mean(entropies)
    
    # Get activation probability (activation count / total tokens)
    activation_count = feature_counts[feat_idx] if feat_idx < len(feature_counts) else 0
    activation_probability = activation_count / total_tokens if total_tokens > 0 else 0
    
    avg_entropies.append(avg_entropy)
    activation_probabilities.append(activation_probability)
    feature_indices.append(feat_idx)

avg_entropies = np.array(avg_entropies)
activation_probabilities = np.array(activation_probabilities)

print(f"Computed entropy for {len(avg_entropies)} features")
print(f"Entropy range: {avg_entropies.min():.4f} to {avg_entropies.max():.4f} bits")
print(f"Activation probability range: {activation_probabilities.min():.6f} to {activation_probabilities.max():.6f}")

# 4. Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(activation_probabilities, avg_entropies, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

plt.xlabel('Activation Probability', fontsize=12, fontweight='bold')
plt.ylabel('Averaged Entropy (bits)', fontsize=12, fontweight='bold')
plt.title(f'Averaged Entropy vs Activation Probability ({SITE})', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add statistics text
stats_text = f'N: {len(avg_entropies)}\nMean entropy: {avg_entropies.mean():.3f} bits\nMean activation prob: {activation_probabilities.mean():.6f}'
plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=10, family='monospace')

plt.tight_layout()
plt.show()

print(f"\nPlot complete for {SITE}")
