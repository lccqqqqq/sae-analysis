import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
SITE = "resid_out_layer3"
BASE_DIR = Path("dictionaries/pythia-70m-deduped") / SITE
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def load_sae(run_dir):
    """Loads the SAE state dict from the specified directory."""
    path = run_dir / "ae.pt"
    if not path.exists():
        raise FileNotFoundError(f"No ae.pt found in {run_dir}")
    
    print(f"[INFO] Loading SAE from {path}")
    sd = torch.load(path, map_location="cpu")
    return sd

def get_sae_weights(sd, d_model):
    """Extracts and shapes encoder/decoder weights from state dict."""
    # Decoder
    dec_w = None
    for k, v in sd.items():
        if "decoder.weight" in k or "dec.weight" in k:
            dec_w = v
            break
    if dec_w is None:
        # Fallback: look for 2D tensor with correct shape
        for v in sd.values():
            if v.ndim == 2 and (v.shape[0] == d_model or v.shape[1] == d_model):
                dec_w = v
                break
    
    if dec_w is None:
        raise ValueError("Could not find decoder weights")

    # Ensure decoder is [d_model, n_latent]
    if dec_w.shape[0] != d_model:
        dec_w = dec_w.T
    
    # Encoder
    enc_w = sd.get("encoder.weight")
    enc_b = sd.get("encoder.bias")
    
    # If encoder weights missing, might be tied weights (transpose of decoder)
    if enc_w is None:
        print("[WARN] No encoder weights found, assuming tied weights (enc = dec.T)")
        enc_w = dec_w.T
    else:
        # Ensure encoder is [d_model, n_latent] for matmul x @ W
        # Usually stored as [n_latent, d_model] in Linear layers
        if enc_w.shape[1] == d_model: 
            enc_w = enc_w.T
        elif enc_w.shape[0] == d_model:
            pass # already correct shape
            
    if enc_b is None:
        enc_b = torch.zeros(dec_w.shape[1])
        
    return {
        "enc_w": enc_w.to(DEVICE),
        "enc_b": enc_b.to(DEVICE),
        "dec_w": dec_w.to(DEVICE),
        "dec_b": sd.get("decoder.bias", torch.zeros(d_model)).to(DEVICE) if sd.get("decoder.bias") is not None else None
    }

def get_model_activations(model, tokenizer, prompt, layer_idx):
    """Runs the model and captures residual stream at the specified layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Hook to capture activation
    activations = []
    def hook_fn(module, input, output):
        # Output of transformer layer is usually (hidden_states, ...)
        if isinstance(output, tuple):
            activations.append(output[0].detach())
        else:
            activations.append(output.detach())
            
    layer = model.gpt_neox.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(**inputs)
        
    handle.remove()
    return activations[0], inputs["input_ids"]

def get_feature_activations(resid, sae_weights):
    """
    Computes feature activations: ReLU(x @ W_enc + b_enc)
    resid: [Batch, Seq, d_model]
    """
    # Encoder forward pass
    # (B, T, d) @ (d, n) + (n) -> (B, T, n)
    x = resid
    enc_w = sae_weights["enc_w"]
    enc_b = sae_weights["enc_b"]
    
    pre_activations = torch.matmul(x, enc_w) + enc_b
    feature_acts = F.relu(pre_activations)
    
    return feature_acts

def decode_feature(feature_idx, sae_weights):
    """Returns the decoder vector for a specific feature index."""
    # dec_w is [d_model, n_latent]
    # We want column feature_idx
    return sae_weights["dec_w"][:, feature_idx]

def plot_heatmap(feature_acts, tokens, tokenizer, threshold=1.0, skip_first=True):
    """
    Plots heatmap of active features above a threshold.
    skip_first: If True, ignores the first token (often an attention sink with high norm) for plotting.
    threshold: Minimum activation value to include a feature.
    """
    # feature_acts: [1, Seq, n_latent] -> remove batch
    acts = feature_acts[0].cpu().numpy() # [Seq, n_latent]
    token_ids = tokens[0].cpu().numpy()
    
    if skip_first and acts.shape[0] > 1:
        print("\n[INFO] Skipping first token for heatmap (Attention Sink / High Norm)")
        acts = acts[1:]
        token_ids = token_ids[1:]

    seq_len = acts.shape[0]
    
    # Find active features for each token
    active_indices = set()
    print(f"\n{'Token':<15} | Features > {threshold} (Index: Activation)")
    print("-" * 60)
    
    token_strs = [tokenizer.decode([t]) for t in token_ids]
    
    for t in range(seq_len):
        # Get indices where activation > threshold
        curr_acts = acts[t]
        active_mask = curr_acts > threshold
        active_idx = np.where(active_mask)[0]
        
        # Sort by activation strength for printing
        # argsort gives ascending, so we take reverse
        sorted_active_idx = active_idx[np.argsort(curr_acts[active_idx])[::-1]]
        
        active_indices.update(active_idx)
        
        # Print info
        token_str = repr(token_strs[t])
        if len(sorted_active_idx) > 0:
            # Print top few for brevity in console, but plot all
            display_limit = 8
            feat_info = ", ".join([f"{idx}:{curr_acts[idx]:.2f}" for idx in sorted_active_idx[:display_limit]])
            if len(sorted_active_idx) > display_limit:
                feat_info += f", ... ({len(sorted_active_idx) - display_limit} more)"
        else:
            feat_info = "<No active features>"
            
        print(f"{token_str:<15} | {feat_info}")

    if len(active_indices) == 0:
        print("\n[WARN] No features found above threshold. Try lowering the threshold.")
        return

    # Filter matrix to only interesting features for plotting
    sorted_indices = sorted(list(active_indices))
    filtered_acts = acts[:, sorted_indices]
    
    plt.figure(figsize=(12, len(sorted_indices) * 0.2 + 4)) # Adjust height based on number of features
    sns.heatmap(filtered_acts.T, cmap="viridis", xticklabels=token_strs, yticklabels=sorted_indices)
    plt.xlabel("Tokens")
    plt.ylabel("Feature Index")
    plt.title(f"SAE Feature Activations (Threshold > {threshold})")
    plt.tight_layout()
    plt.savefig("sae_heatmap.png")
    print("\n[INFO] Heatmap saved to sae_heatmap.png")

def interpret_feature(feature_idx, sae_weights, model, tokenizer, top_k=10):
    """
    Interprets a feature by projecting it onto the vocabulary (Logit Lens).
    Filters out garbage tokens (containing replacement character ).
    """
    # 1. Get the feature vector (direction in residual stream)
    # Shape: [d_model]
    feature_vec = decode_feature(feature_idx, sae_weights)
    
    # 2. Get the Unembedding Matrix (W_U)
    # Pythia/GPT-NeoX usually stores this in embed_out
    if hasattr(model, "embed_out"):
        W_U = model.embed_out.weight # [vocab_size, d_model]
    else:
        print("[WARN] Could not find embed_out, trying get_output_embeddings()")
        W_U = model.get_output_embeddings().weight
        
    # 3. Compute Logits: v @ W_U.T
    # [d_model] @ [d_model, vocab_size] -> [vocab_size]
    logits = torch.matmul(feature_vec, W_U.T)
    
    # 4. Get Top-K Tokens (fetch more to allow for filtering)
    # Fetch top 100 to ensure we find enough valid tokens
    top_vals, top_idxs = torch.topk(logits, k=100)
    
    print(f"\n[INFO] Logit Lens for Feature {feature_idx}:")
    print(f"{'Token':<15} | {'Logit':<10}")
    print("-" * 30)
    
    count = 0
    for score, idx in zip(top_vals, top_idxs):
        token_str = tokenizer.decode([idx])
            
        print(f"{repr(token_str):<15} | {score:.4f}")
        count += 1
        if count >= top_k:
            break

def main():
    # 1. Find Run
    run_dir = None
    for p in BASE_DIR.iterdir():
        if p.is_dir() and (p / "ae.pt").exists():
            run_dir = p
            break
    if not run_dir:
        raise FileNotFoundError("No run directory found")
        
    # 2. Load Model & SAE
    print("[INFO] Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    d_model = model.config.hidden_size
    
    sae_sd = load_sae(run_dir)
    sae_weights = get_sae_weights(sae_sd, d_model)
    
    # 3. Run Inference
    prompt = "I want to understand transformer models by sparse-auto-encoder"
    print(f"\n[INFO] Processing prompt: {repr(prompt)}")
    
    # Parse layer index from SITE
    layer_idx = int(SITE.rsplit("layer", 1)[-1])
    
    resid, tokens = get_model_activations(model, tokenizer, prompt, layer_idx)
    print(f"[DEBUG] Residual shape: {resid.shape}")
    
    # --- Debugging High Activations ---
    print("\n[DEBUG] Token Norms & Identity:")
    token_ids = tokens[0].cpu().numpy()
    norms = torch.norm(resid[0], dim=-1).cpu().numpy()
    for i, (tid, norm) in enumerate(zip(token_ids, norms)):
        t_str = tokenizer.decode([tid])
        print(f"Pos {i}: '{t_str}' (ID: {tid}) | Norm: {norm:.4f}")
    # ----------------------------------
    
    # 4. Get SAE Activations
    feature_acts = get_feature_activations(resid, sae_weights)
    print(f"[DEBUG] Feature activations shape: {feature_acts.shape}")
    
    # 5. Visualize
    plot_heatmap(feature_acts, tokens, tokenizer, threshold=0.5, skip_first=True)
    
    # 6. Demo Decoding
    # Let's decode the max feature from the last token
    last_token_acts = feature_acts[0, -1]
    max_feat_idx = torch.argmax(last_token_acts).item()
    print(f"\n[INFO] Decoding strongest feature at last token: Index {max_feat_idx}")
    
    decoded_vec = decode_feature(max_feat_idx, sae_weights)
    print(f"[DEBUG] Decoded vector shape: {decoded_vec.shape}")
    print(f"[DEBUG] First 5 values: {decoded_vec[:5].tolist()}")

    # 7. Interpret Feature
    interpret_feature(max_feat_idx, sae_weights, model, tokenizer)

if __name__ == "__main__":
    main()
