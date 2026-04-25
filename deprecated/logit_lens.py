import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    print(f"[INFO] Loading Model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    prompt = """
    To be or not to be, """
    print(f"[INFO] Processing prompt: {repr(prompt)}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"][0]
    tokens = [tokenizer.decode([t]) for t in input_ids]
    
    # Run model with output_hidden_states=True
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # hidden_states is a tuple of (layer_0_input, layer_0_output, ..., layer_N_output)
    # For Pythia (GPT-NeoX), usually:
    # Index 0: Embeddings (Input to Layer 0)
    # Index 1..N: Output of each layer
    # Last Index: Final output before LN_f (usually)
    
    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states)
    
    print(f"\n[INFO] Logit Lens Analysis ({num_layers} layers captured)")
    print("-" * 100)
    
    # Header
    header = f"{'Layer':<6} | " + " | ".join([f"{t:<12}" for t in tokens])
    print(header)
    print("-" * len(header))
    
    # Iterate through layers
    for i, layer_hidden in enumerate(hidden_states):
        # layer_hidden: [1, Seq, d_model]
        
        # CRITICAL: Apply Final Layer Norm
        # The unembedding matrix expects normalized inputs.
        # Intermediate layers are NOT normalized by the final LN yet.
        normed = model.gpt_neox.final_layer_norm(layer_hidden)
        
        # Unembed: [1, Seq, d] @ [vocab, d].T -> [1, Seq, vocab]
        if hasattr(model, "embed_out"):
            logits = torch.matmul(normed, model.embed_out.weight.T)
        else:
            logits = torch.matmul(normed, model.get_output_embeddings().weight.T)
            
        # Get max token
        preds = torch.argmax(logits[0], dim=-1) # [Seq]
        
        pred_tokens = []
        for t_idx in preds:
            t_str = tokenizer.decode([t_idx])
            # Clean up newlines/spaces for table
            t_str = t_str.replace("\n", "\\n").strip()
            if len(t_str) > 12:
                t_str = t_str[:9] + "..."
            pred_tokens.append(t_str)
            
        row_str = f"L{i:<5} | " + " | ".join([f"{t:<12}" for t in pred_tokens])
        print(row_str)

if __name__ == "__main__":
    main()
