# lm_infer.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "EleutherAI/pythia-70m-deduped"

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

prompt = "Is the dog white or black? "
inputs = tok(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        eos_token_id=tok.eos_token_id,
    )

print(tok.decode(out[0], skip_special_tokens=True))
