# Plot averaged entropy vs activation count — layer 5 variant
# Same as plot_entropy_vs_activation.py but reads layer 5 data instead of layer 3.

import torch
import numpy as np
import matplotlib.pyplot as plt

# Configuration
SITE = "resid_out_layer5"  # Layer 5 instead of the default layer 3

# Load both data files
print(f"Loading data for {SITE}...")
influence_data = torch.load(f'feature_token_influence_{SITE}.pt', map_location='cpu', weights_only=False)
feature_influences = influence_data['feature_influences']
config = influence_data.get('config', {})
BATCH_SIZE = config.get('batch_size', 64)

sparsity_data = torch.load(f'feature_sparsity_data_{SITE}.pt', map_location='cpu', weights_only=False)
frequencies = sparsity_data['frequencies']

# For each tracked feature compute (activation_probability, mean_entropy_across_batches).
def shannon_entropy_of_norms(norms):
    eps = 1e-12
    P = (norms + eps) / (np.sum(norms) + eps)
    P = P[P > 0]
    return float(-(P * np.log2(P)).sum())

xs, ys = [], []
for feat_idx, fdata in feature_influences.items():
    all_inf = fdata.get('all_influences', [])
    if not all_inf:
        continue
    entropies = [shannon_entropy_of_norms(np.asarray(arr)) for arr in all_inf]
    xs.append(float(frequencies[feat_idx]))
    ys.append(float(np.mean(entropies)))

xs = np.array(xs)
ys = np.array(ys)
print(f"Activation probability range: {xs.min():.6f} to {xs.max():.6f}")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(xs, ys, alpha=0.6, s=18)
ax.set_xscale('log')
ax.set_xlabel('Activation Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Averaged Entropy (bits)', fontsize=12, fontweight='bold')
ax.set_title(f'Averaged Entropy vs Activation Probability ({SITE})', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(f"\nPlot complete for {SITE}")
