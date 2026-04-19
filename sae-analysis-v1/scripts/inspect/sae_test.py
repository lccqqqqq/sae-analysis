# sae_test_decode_only.py
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Try common key names for decoder weight/bias
dec_w = None
dec_b = None
for k, v in sd.items():
    lk = k.lower()
    if dec_w is None and ("decoder.weight" in lk or "dec.weight" in lk):
        dec_w = v
    if dec_b is None and ("decoder.bias" in lk or "dec.bias" in lk):
        dec_b = v
if dec_w is None:
    # fallback: take first 2D weight
    for k, v in sd.items():
        if v.ndim == 2:
            dec_w = v
            break
if dec_w is None:
    raise RuntimeError("Could not find decoder weight in ae.pt")

print(f"[DEBUG] Decoder weight shape: {dec_w.shape}")
if dec_b is not None:
    print(f"[DEBUG] Decoder bias shape: {dec_b.shape}")
else:
    print("[DEBUG] No decoder bias found")

# ---- Load LM and prompt ----
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

prompt = "quantum mechanics and gravity "
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

# ---- Build ONE decoded feature at the last position
col_norms = torch.linalg.vector_norm(dec_W, dim=0)  # [n_latent]
j = int(torch.argmax(col_norms).item())             # pick a strong feature
mag = 1.0                                           # scale as you wish
print(f"[DEBUG] Selected feature index: {j}, magnitude: {mag}")

B, T = enc["input_ids"].shape
n_latent = dec_W.shape[1]
feats = torch.zeros((B, T, n_latent), device=device)
feats[0, -1, j] = mag

# Decode to residual space: [B,T,n] @ [d,n]^T -> [B,T,d]
patched_resid = torch.einsum("btn, dn -> btd", feats, dec_W)
if dec_b is not None:
    patched_resid = patched_resid + dec_b.view(1, 1, -1)
print(f"[DEBUG] Patched residual shape: {patched_resid.shape}")

# Sanity: shapes must match the layer output
assert patched_resid.shape == expected_shape, f"Patched {patched_resid.shape} != expected {expected_shape}"

# ---- Hook: add the patched residual to the layer output at the specified position
try:
    target_layer = lm.gpt_neox.layers[layer_idx]
    print(f"[DEBUG] Target layer: {target_layer.__class__.__name__}")
except (AttributeError, IndexError) as e:
    print(f"[DEBUG] Error accessing layer: {e}")
    print(f"[DEBUG] Model structure: {lm.__class__.__name__}")
    print(f"[DEBUG] Available attributes: {dir(lm)}")
    raise

# Store the patched residual for the hook to use
patched_addition = torch.zeros_like(patched_resid)
patched_addition[0, -1, :] = patched_resid[0, -1, :]  # Only modify the last token

def add_patched_residual(_module, _inp, _out):
    # Handle tuple outputs from transformer layers
    if isinstance(_out, tuple):
        # The first element is usually the hidden states
        hidden_states = _out[0]
        modified_hidden_states = hidden_states + patched_addition
        # Return the tuple with modified hidden states
        return (modified_hidden_states,) + _out[1:]
    else:
        # If it's just a tensor, add directly
        return _out + patched_addition

handle = target_layer.register_forward_hook(add_patched_residual)
with torch.no_grad():
    patched_out = lm(enc["input_ids"])
handle.remove()

patched_logits = patched_out.logits[0, -1].float()

# ---- Analyze changes
delta = patched_logits - base_logits
topk = torch.topk(delta, k=15)
print("Top +Δ logits (tokens boosted by the single decoded feature):")
for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
    print(f"{tok.decode([idx])!r:>12} : {score:.4f}")

p = torch.softmax(patched_logits, dim=-1)
q = torch.softmax(base_logits, dim=-1)
kl = torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).item()
print(f"\nKL(patched || base) @ last position: {kl:.6f}")
print(f"[Info] SITE={SITE}, layer={layer_idx}, feature j={j}, mag={mag}, run={run_dir.name}")
