"""Sanity-check that local Gemma+SAE activations match Neuronpedia's.

For each feature in the YAML, takes the first event from the adapter-built
top_contexts JSON, runs a local forward pass through the model + SAE, and
compares the local SAE feature value at the peak position against
Neuronpedia's reported ``activation`` (= ``maxValue`` from the API).

A small absolute and relative tolerance is allowed; failures (>5 % rel.
error) abort with a non-zero exit code so the chained submit script
won't issue the parallel entropy jobs against bad data.

Used by ``scripts/experiments/submit_cherry_picked_entropy_neuronpedia.sh``
as the smoke step between the CPU adapter and the parallel cluster jobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_adapters import get_layer, load_model
from presets import get_preset
from sae_adapters import load_sae

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def run(yaml_path: Path, top_contexts_path: Path, corpus_tensor_path: Path,
        rel_tol: float, abs_tol: float) -> int:
    cfg = yaml.safe_load(yaml_path.read_text())
    preset = get_preset(cfg["preset"])
    layer_idx = int(cfg["layer"])

    model, _ = load_model(preset, DEVICE)
    sae = load_sae(preset, layer_idx, DEVICE)
    print(f"[INFO] preset={preset.name} layer={layer_idx} arch={sae.arch}",
          flush=True)

    corpus_tokens = torch.load(corpus_tensor_path).to(torch.long).to(DEVICE)
    top_contexts = json.loads(top_contexts_path.read_text())
    print(f"[INFO] corpus_tokens shape={tuple(corpus_tokens.shape)}",
          flush=True)

    # Hook the residual stream at layer_idx.
    captured: list[torch.Tensor] = []

    def hook_fn(module, inputs, output):  # noqa: ARG001
        captured.append(output[0] if isinstance(output, tuple) else output)

    layer = get_layer(model, preset, layer_idx)
    handle = layer.register_forward_hook(hook_fn)

    n_pass = 0
    n_fail = 0
    rows = []
    try:
        for fid_str, payload in top_contexts.items():
            fid = int(fid_str)
            examples = payload.get("examples", [])
            if not examples:
                continue
            ex = examples[0]
            abs_peak = int(ex["window_start_token"]) + int(ex["token_idx"])
            ws = int(ex["window_start_token"])
            event_end = ws + len(ex["tokens"])
            window = corpus_tokens[ws:event_end].unsqueeze(0)
            captured.clear()
            with torch.no_grad():
                _ = model(window)
            resid = captured[0]                          # [1, T, d_model]
            feats = sae.encode(resid)                    # [1, T, n_latent]
            local_act = float(feats[0, int(ex["token_idx"]), fid])
            np_act = float(ex["activation"])
            diff = abs(local_act - np_act)
            rel = diff / max(abs(np_act), 1e-9)
            ok = (diff <= abs_tol) or (rel <= rel_tol)
            if ok:
                n_pass += 1
            else:
                n_fail += 1
            rows.append((fid, np_act, local_act, diff, rel, ok))
            tag = "OK " if ok else "FAIL"
            print(f"  [{tag}] feat={fid:>5d}  np={np_act:>8.3f}  "
                  f"local={local_act:>8.3f}  diff={diff:>7.3f}  "
                  f"rel={rel*100:>6.2f}%", flush=True)
    finally:
        handle.remove()

    print(f"\n[VERIFY] pass={n_pass}  fail={n_fail}  "
          f"abs_tol={abs_tol}  rel_tol={rel_tol}", flush=True)
    if n_fail > 0:
        print("[ERROR] activation match failed for some features; "
              "DO NOT submit entropy jobs against this corpus tensor.",
              flush=True)
        return 1
    print("[OK] all features matched within tolerance.", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", type=Path, required=True)
    ap.add_argument("--top-contexts", type=Path, required=True)
    ap.add_argument("--corpus-tensor", type=Path, required=True)
    ap.add_argument("--rel-tol", type=float, default=0.05,
                    help="Allowed relative error in the activation match.")
    ap.add_argument("--abs-tol", type=float, default=0.5,
                    help="Allowed absolute error (overrides rel-tol when "
                         "the activation is near zero).")
    args = ap.parse_args()
    sys.exit(run(args.yaml, args.top_contexts, args.corpus_tensor,
                 args.rel_tol, args.abs_tol))


if __name__ == "__main__":
    main()
