import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    print(f"[INFO] Loading Model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    prompt = "Entropy measures the amount of information in a message, "
    print(f"[INFO] Prompt: {repr(prompt)}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    print("\n[INFO] Generating...")
    # Generate up to 50 new tokens
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=True,      # Enable sampling for more interesting text
        temperature=0.7,     # Control randomness
        pad_token_id=tokenizer.eos_token_id # Suppress warning
    )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("-" * 60)
    print(generated_text)
    print("-" * 60)

if __name__ == "__main__":
    main()
