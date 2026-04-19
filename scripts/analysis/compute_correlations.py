import torch
from transformers import AutoTokenizer
from pathlib import Path
from collections import Counter
import sys

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DATA_FILE = Path("wikitext-2-train.txt")
MIN_UNIQUE_TOKENS = 300

def main(site=None):
    """
    Main function to compute correlation matrix for a given layer.
    
    Args:
        site: Site string like "resid_out_layer3" (if None, uses default)
    """
    # Use provided site or default
    if site is None:
        site = "resid_out_layer0"
    
    pt_file = Path(f"feature_sparsity_data_{site}.pt")
    output_file = Path(f"correlation_matrix_{site}.pt")
    
    if not pt_file.exists():
        print(f"Error: {pt_file} not found.")
        return

    print(f"\n{'='*60}")
    print(f"[INFO] Processing {site}")
    print(f"{'='*60}")
    
    print(f"Loading {pt_file}...")
    data = torch.load(pt_file, map_location="cpu")
    
    feature_counts = data["feature_counts"]
    # frequencies = data["frequencies"] # Not strictly needed if we compute from counts
    total_tokens_processed = data["total_tokens"]
    feature_token_counts = data["feature_token_counts"] # List of Counters
    
    n_latent = len(feature_token_counts)
    print(f"Total features: {n_latent}")
    print(f"Total tokens processed in data: {total_tokens_processed}")

    # 1. Get Global Token Counts (N(a))
    # We need to re-tokenize the text to get the exact counts of each token in the processed subset.
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found. Cannot compute global token counts.")
        return
        
    print(f"Loading and tokenizing {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
        
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    
    # Slice to the same length as used in feature_sparsity.py
    if tokens.shape[0] > total_tokens_processed:
        print(f"Slicing tokens to first {total_tokens_processed}...")
        tokens = tokens[:total_tokens_processed]
    elif tokens.shape[0] < total_tokens_processed:
        print(f"Warning: Tokenized text has fewer tokens ({tokens.shape[0]}) than recorded in .pt file ({total_tokens_processed}). Using available tokens.")
        # This might cause issues if the .pt file was generated from a different text source or length.
        # But we'll proceed with what we have.
    
    # Count global token occurrences
    print("Counting global token frequencies...")
    global_token_counts = Counter(tokens.tolist())
    
    # 2. Identify Leading Features
    # "triggered by more than 1000 unique tokens"
    print("Identifying leading features...")
    leading_features = []
    for i in range(n_latent):
        unique_tokens = len(feature_token_counts[i])
        if unique_tokens > MIN_UNIQUE_TOKENS:
            leading_features.append(i)
            
    print(f"Found {len(leading_features)} leading features (triggered by > {MIN_UNIQUE_TOKENS} unique tokens).")
    
    if len(leading_features) == 0:
        print("No leading features found.")
        return

    # 3. Compute Correlations
    # Formula: <A_i A_j> - <A_i><A_j>
    # Assumption: A_i(a) = P(feature i active | token a) = N(i, a) / N(a)
    # <A_i> = sum_a P(a) A_i(a) = sum_a (N(a)/T) * (N(i, a)/N(a)) = sum_a N(i, a) / T = N_i / T
    # <A_i A_j> = sum_a P(a) A_i(a) A_j(a) = sum_a (N(a)/T) * (N(i, a)/N(a)) * (N(j, a)/N(a))
    #           = (1/T) * sum_a [ N(i, a) * N(j, a) / N(a) ]
    
    # We will compute the covariance matrix for the leading features.
    # To do this efficiently, we can construct a sparse matrix or just iterate.
    # Since we need pairwise correlations between all leading features, let's see how many there are.
    # If there are too many, we might need a more efficient approach.
    
    # Let's construct a matrix M where M[i, a] = A_i(a) * sqrt(P(a)) ?
    # Cov_ij = <A_i A_j> - <A_i><A_j>
    #        = sum_a P(a) A_i(a) A_j(a) - ...
    # Let v_i be a vector of size |V| (vocab size) where v_i[a] = A_i(a) * sqrt(P(a))
    # Then <A_i A_j> = dot(v_i, v_j)
    # P(a) = N(a) / T
    # A_i(a) = N(i, a) / N(a)
    # v_i[a] = (N(i, a) / N(a)) * sqrt(N(a)/T) = N(i, a) / sqrt(N(a) * T)
    
    # So we can construct a sparse matrix V of shape (num_leading_features, vocab_size).
    # V[k, a] = N(leading_features[k], a) / sqrt(N(a) * T)
    # Then Cov_matrix = V @ V.T - mean_vec @ mean_vec.T
    # where mean_vec[k] = <A_leading_features[k]> = N(leading_features[k]) / T
    
    vocab_size = tokenizer.vocab_size # Or max token id + 1
    # Actually, let's use the max token id found in data to be safe, or just vocab_size.
    # global_token_counts keys are token ids.
    
    # Map token_id to index in our vector? Or just use token_id as index.
    # Vocab size is around 50k for Pythia.
    
    print("Constructing feature vectors...")
    
    # We need a dense or sparse matrix. 
    # num_leading_features x vocab_size
    # If num_leading_features is small (e.g. < 1000), dense might be fine.
    # If large, maybe sparse.
    
    # Let's check how many leading features we have.
    # If it's huge, we might run out of memory with dense.
    # But let's try dense first or use torch.sparse.
    
    T = total_tokens_processed
    
    # Precompute sqrt(N(a) * T) for all a
    # We only care about tokens that appear in the dataset.
    
    # Create a tensor for normalization
    # norm[a] = 1 / sqrt(N(a) * T)
    # If N(a) is 0, it shouldn't happen for tokens in the dataset.
    
    # Use a dictionary for sparse construction
    indices = []
    values = []
    
    # Also compute means
    means = torch.zeros(len(leading_features))
    
    for idx, f_idx in enumerate(leading_features):
        cntr = feature_token_counts[f_idx]
        total_f_count = feature_counts[f_idx].item()
        means[idx] = total_f_count / T
        
        for t_id, count in cntr.items():
            # count is N(i, a)
            # global_count is N(a)
            global_count = global_token_counts.get(t_id, 0)
            if global_count == 0:
                # This implies the token was in feature counts but not in our re-tokenized data.
                # This is a mismatch.
                continue
                
            val = count / ( (global_count * T)**0.5 )
            indices.append([idx, t_id])
            values.append(val)
            
    indices = torch.tensor(indices).t() # [2, nnz]
    values = torch.tensor(values)
    
    # Shape: [num_leading, vocab_size]
    # We need to know the max token id to set shape
    max_token_id = max(global_token_counts.keys()) if global_token_counts else 0
    # Ensure shape is large enough
    vocab_dim = max_token_id + 1
    
    print(f"Matrix shape: ({len(leading_features)}, {vocab_dim})")
    
    V = torch.sparse_coo_tensor(indices, values, (len(leading_features), vocab_dim))
    
    # Compute Gram matrix: G = V @ V.T
    # Sparse matrix multiplication: sparse @ sparse_T -> dense (usually)
    # Or sparse @ dense.
    # PyTorch sparse matmul support is limited.
    # V is sparse. V.T is sparse.
    # torch.sparse.mm(V, V.t()) might work if V is coalesced.
    
    V = V.coalesce()
    
    print("Computing covariance matrix...")
    # If V is too large, this might be slow.
    # Try converting to dense if it fits in memory?
    # len(leading) * vocab_dim * 4 bytes.
    # If len(leading) = 1000, vocab = 50k -> 50M floats -> 200MB. Fits easily.
    
    if len(leading_features) * vocab_dim < 1e8: # < 400MB
        V_dense = V.to_dense()
        Cov = torch.matmul(V_dense, V_dense.t())
    else:
        # Fallback to sparse mm if possible
        try:
            Cov = torch.sparse.mm(V, V.t())
        except Exception as e:
            print(f"Sparse MM failed: {e}. Trying dense chunked or loop.")
            # Fallback: just use dense if memory allows, or loop.
            V_dense = V.to_dense()
            Cov = torch.matmul(V_dense, V_dense.t())

    # Subtract means product
    # Cov_ij = <A_i A_j> - <A_i><A_j>
    # M_ij = <A_i><A_j>
    Means_outer = torch.outer(means, means)
    
    Cov = Cov - Means_outer
    
    print("Computation complete.")
    
    # Show top correlations (positive and negative)
    # We want to find pairs (i, j) with high absolute covariance (or correlation?)
    # The user asked for "correlation defined by <A_iA_j>-<A_i><A_j>", which is covariance.
    # But usually people want to see the most related features.
    # Let's print the top 20 pairs by absolute covariance.
    
    # Mask diagonal for printing top pairs, but keep original Cov for saving
    Cov_for_printing = Cov.clone()
    diag_mask = torch.eye(len(leading_features), dtype=torch.bool)
    Cov_for_printing.masked_fill_(diag_mask, 0)
    
    # Flatten and topk
    # We only care about upper triangle to avoid duplicates
    triu_indices = torch.triu_indices(len(leading_features), len(leading_features), offset=1)
    # This might be large if many leading features.
    
    # Let's just take topk from the flattened matrix and filter duplicates.
    top_vals, top_flat_indices = torch.topk(Cov_for_printing.abs().flatten(), k=40)
    
    print("\nTop Feature Pairs by Covariance (<A_i A_j> - <A_i><A_j>):")
    print(f"{'Feat 1':<10} | {'Feat 2':<10} | {'Covariance':<12}")
    
    seen_pairs = set()
    
    for val, flat_idx in zip(top_vals, top_flat_indices):
        idx1 = (flat_idx // len(leading_features)).item()
        idx2 = (flat_idx % len(leading_features)).item()
        
        if idx1 > idx2: idx1, idx2 = idx2, idx1
        if (idx1, idx2) in seen_pairs: continue
        seen_pairs.add((idx1, idx2))
        
        real_val = Cov[idx1, idx2].item()
        f1 = leading_features[idx1]
        f2 = leading_features[idx2]
        
        print(f"{f1:<10} | {f2:<10} | {real_val:.2e}")

    print(f"Saving covariance matrix to {output_file}...")
    torch.save({
        "covariance_matrix": Cov,
        "leading_features": leading_features,
        "site": site
    }, output_file)
    print(f"✓ Successfully completed {site}\n")

if __name__ == "__main__":
    # Option 1: Run for a single layer (backward compatible)
    # main(site="resid_out_layer0")
    
    # Option 2: Run for multiple layers
    sites = [
        "resid_out_layer0",
        "resid_out_layer1",
        "resid_out_layer2",
        "resid_out_layer3",
        "resid_out_layer4",
        "resid_out_layer5",
    ]
    
    for site in sites:
        try:
            main(site=site)
        except Exception as e:
            print(f"[ERROR] Failed to process {site}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("All layers processed!")
    print("="*60)
