# sae_test_decode_only.py
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_NAME = "EleutherAI/pythia-70m-deduped"
SITE = "resid_out_layer3"  # e.g., resid_out_layerK / mlp_out_layerK / attn_out_layerK
BASE = Path("dictionaries/pythia-70m-deduped") / SITE

# --- find a run folder that contains ae.pt ---
run_dir = None
for p in BASE.iterdir():
    if p.is_dir() and (p / "ae.pt").exists():
        run_dir = p
        break
if run_dir is None:
    raise FileNotFoundError(f"No ae.pt found under {BASE}")

print(f"[DEBUG] Found run directory: {run_dir}")
sd = torch.load(run_dir / "ae.pt", map_location="cpu")
print(f"[DEBUG] Loaded state dict with keys: {list(sd.keys())}")

# ---- Grab decoder (as before) ----
dec_w = None
dec_b = None
for k, v in sd.items():
    lk = k.lower()
    if dec_w is None and ("decoder.weight" in lk or "dec.weight" in lk):
        dec_w = v
    if dec_b is None and ("decoder.bias" in lk or "dec.bias" in lk):
        dec_b = v
if dec_w is None:
    for k, v in sd.items():
        if v.ndim == 2:
            dec_w = v
            break
if dec_w is None:
    raise RuntimeError("Could not find decoder weight in ae.pt")
print(f"[DEBUG] Decoder weight shape: {dec_w.shape}")
print("[DEBUG] Decoder bias found" if dec_b is not None else "[DEBUG] No decoder bias found")

# ---- NEW: Grab encoder for input-dependent feature selection ----
enc_w = None
enc_b = None
for k, v in sd.items():
    lk = k.lower()
    if enc_w is None and ("encoder.weight" in lk or "enc.weight" in lk):
        enc_w = v
    if enc_b is None and ("encoder.bias" in lk or "enc.bias" in lk):
        enc_b = v
if enc_w is None:
    raise RuntimeError("Could not find encoder weight in ae.pt")
print(f"[DEBUG] Encoder weight shape: {enc_w.shape}")
if enc_b is not None:
    print(f"[DEBUG] Encoder bias shape: {enc_b.shape}")

# ---- Load LM and prompt ----
tok = AutoModelForCausalLM.from_pretrained  # type: ignore  # silence pylance for the next line
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
lm  = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
lm.eval()
d_model = lm.config.hidden_size  # 512 for Pythia-70M
print(f"[DEBUG] Model d_model: {d_model}")

def as_decoder(weight, d_model):
    # ensure shape [d_model, n_latent]
    return weight if weight.shape[0] == d_model else weight.T

dec_W = as_decoder(dec_w, d_model).to(device)  # [d_model, n_latent]
dec_b = (dec_b.to(device) if dec_b is not None else None)
enc_W = enc_w.to(device)                        # usually [n_latent, d_model]
enc_b = (enc_b.to(device) if enc_b is not None else None)

prompt = "What are the important questions in high temperature superconductivity? "
enc = tok(prompt, return_tensors="pt").to(device)

# Layer index from folder name
assert "layer" in SITE and SITE.rsplit("layer", 1)[-1].isdigit(), f"Cannot parse layer index from SITE={SITE}"
layer_idx = int(SITE.rsplit("layer", 1)[-1])
print(f"[DEBUG] Layer index: {layer_idx}")

# ---- Baseline forward (also gives us expected shapes)
with torch.no_grad():
    base_out = lm(enc["input_ids"], output_hidden_states=True)
base_logits = base_out.logits[0, -1].float()
expected_shape = base_out.hidden_states[layer_idx + 1].shape  # [B, T, d_model]
print(f"[DEBUG] Expected shape: {expected_shape}")

# ---- NEW: Encoder pass to pick input-dependent feature (last token)
with torch.no_grad():
    resid = base_out.hidden_states[layer_idx + 1].to(device)   # [B, T, d_model]
    # feats = resid @ enc_W^T (+ enc_b)
    # enc_W is [n_latent, d_model] -> einsum "btd, nd -> btn"
    feats_full = torch.einsum("btd, nd -> btn", resid, enc_W)
    if enc_b is not None:
        feats_full = feats_full + enc_b.view(1, 1, -1)

# Select the top-activated feature at the last token
j = int(torch.argmax(feats_full[0, -1]).item())
mag = float(feats_full[0, -1, j].item())
print(f"[DEBUG] Selected feature index (encoder-based): {j}, magnitude: {mag:.6f}")

# ---- Build ONE decoded feature at the last position (same as before)
B, T = enc["input_ids"].shape
n_latent = dec_W.shape[1]
feats = torch.zeros((B, T, n_latent), device=device)
feats[0, -1, j] = mag  # use the encoder-derived magnitude

# Decode to residual space: [B,T,n] @ [d,n]^T -> [B,T,d]
patched_resid = torch.einsum("btn, dn -> btd", feats, dec_W)
if dec_b is not None:
    patched_resid = patched_resid + dec_b.view(1, 1, -1)
print(f"[DEBUG] Patched residual shape: {patched_resid.shape}")

# Sanity: shapes must match the layer output
assert patched_resid.shape == expected_shape, f"Patched {patched_resid.shape} != expected {expected_shape}"

# ---- Hook: add the patched residual to the layer output at the specified position (unchanged pattern)
try:
    target_layer = lm.gpt_neox.layers[layer_idx]
    print(f"[DEBUG] Target layer: {target_layer.__class__.__name__}")
except (AttributeError, IndexError) as e:
    print(f"[DEBUG] Error accessing layer: {e}")
    print(f"[DEBUG] Model structure: {lm.__class__.__name__}")
    print(f"[DEBUG] Available attributes: {dir(lm)}")
    raise

# Only modify the last token position
patched_addition = torch.zeros_like(patched_resid)
patched_addition[0, -1, :] = patched_resid[0, -1, :]

def add_patched_residual(_module, _inp, _out):
    if isinstance(_out, tuple):
        hidden_states = _out[0]
        modified_hidden_states = hidden_states + patched_addition
        return (modified_hidden_states,) + _out[1:]
    else:
        return _out + patched_addition

handle = target_layer.register_forward_hook(add_patched_residual)
with torch.no_grad():
    patched_out = lm(enc["input_ids"])
handle.remove()

patched_logits = patched_out.logits[0, -1].float()

# ---- Analyze changes (unchanged)
delta = patched_logits - base_logits
topk = torch.topk(delta, k=15)
print("Top +Δ logits (tokens boosted by the single decoded feature):")
for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
    print(f"{tok.decode([idx])!r:>12} : {score:.4f}")

p = torch.softmax(patched_logits, dim=-1)
q = torch.softmax(base_logits, dim=-1)
kl = torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).item()
print(f"\nKL(patched || base) @ last position: {kl:.6f}")
print(f"[Info] SITE={SITE}, layer={layer_idx}, feature j={j}, mag={mag:.6f}, run={run_dir.name}")

# ---- Plot logits sorted by activation ----
def plot_logits_comparison(base_logits, patched_logits, tokenizer, top_k=50, save_path=None):
    """
    Plot comparison of base vs patched logits, sorted by activation.
    
    Args:
        base_logits: Baseline logits tensor
        patched_logits: Patched logits tensor  
        tokenizer: Tokenizer for decoding token IDs
        top_k: Number of top tokens to show
        save_path: Optional path to save the plot
    """
    # Convert to numpy for easier handling
    base_np = base_logits.cpu().numpy()
    patched_np = patched_logits.cpu().numpy()
    delta_np = patched_np - base_np
    
    # Get top-k tokens by patched logits
    top_indices = np.argsort(patched_np)[-top_k:][::-1]
    
    # Get token strings (handle special tokens gracefully)
    token_strs = []
    for idx in top_indices:
        try:
            token_str = tokenizer.decode([idx])
            # Clean up the token string for display
            if token_str.startswith(' '):
                token_str = '▁' + token_str[1:]  # Use ▁ to represent space
            elif token_str == '':
                token_str = '<empty>'
            token_strs.append(f"{token_str} ({idx})")
        except:
            token_strs.append(f"<unk> ({idx})")
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    x_pos = np.arange(len(top_indices))
    
    # Plot 1: Base logits
    ax1.bar(x_pos, base_np[top_indices], alpha=0.7, color='blue')
    ax1.set_title('Base Logits (Top tokens by patched activation)', fontsize=14)
    ax1.set_ylabel('Logit Value')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(token_strs, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Patched logits
    ax2.bar(x_pos, patched_np[top_indices], alpha=0.7, color='red')
    ax2.set_title('Patched Logits (Top tokens by patched activation)', fontsize=14)
    ax2.set_ylabel('Logit Value')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(token_strs, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Delta (change in logits)
    colors = ['green' if d > 0 else 'orange' for d in delta_np[top_indices]]
    ax3.bar(x_pos, delta_np[top_indices], alpha=0.7, color=colors)
    ax3.set_title('Δ Logits (Patched - Base)', fontsize=14)
    ax3.set_ylabel('Δ Logit Value')
    ax3.set_xlabel('Tokens')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(token_strs, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig

# Generate the plot
print("\n[DEBUG] Generating logits comparison plot...")
fig = plot_logits_comparison(
    base_logits, 
    patched_logits, 
    tok, 
    top_k=30,  # Show top 30 tokens
    save_path=f"logits_comparison_layer{layer_idx}_feature{j}.png"
)
