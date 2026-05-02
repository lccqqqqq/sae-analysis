"""Top-k activating context windows for a list of SAE feature indices.

For each requested feature, this scans the corpus in fixed-length windows
and keeps the top-k (window, position) pairs with the highest activation.
For each such pair we record the surrounding token strings so the user can
sanity-check the feature's behaviour and assign a tier label
(token / phrase / concept / abstract).

Output: one JSON file at ``data/<preset>/top_contexts_layer<L>.json`` with

    {
      "<feature_id>": {
        "max_activation": float,
        "examples": [
          {
            "activation": float,
            "token_idx": int,             # position within the window
            "tokens": [str, ...],         # the full window, decoded
            "activating_token": str,
            "window_start_token": int,    # absolute index in the corpus
          },
          ...
        ]
      },
      ...
    }

A pretty-printed companion file ``top_contexts_layer<L>.txt`` is emitted
for terminal inspection.

Used as Step 0 of ``scripts/experiments/submit_cherry_picked_entropy.sh``
and as a standalone tool for browsing candidate features before adding
them to a cherry-pick YAML.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_wikitext_train_text
from model_adapters import get_layer, load_model
from presets import get_preset
from sae_adapters import load_sae

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def collect_top_contexts(
    feature_ids: list[int],
    preset_name: str,
    layer_idx: int,
    top_k: int = 8,
    context_len: int = 64,
    max_batches: int | None = 4000,
    log_every: int = 200,
) -> dict:
    """Return the top-k activations per feature, with surrounding tokens."""
    preset = get_preset(preset_name)
    model, tokenizer = load_model(preset, DEVICE)
    sae = load_sae(preset, layer_idx, DEVICE)
    feature_ids_t = torch.tensor(feature_ids, dtype=torch.long, device=DEVICE)
    feat_pos = {fid: i for i, fid in enumerate(feature_ids)}

    # Per-feature min-heap of (activation, batch_id, position_in_window).
    # Heap stores up to top_k items; smallest at root so we can pop on overflow.
    heaps: dict[int, list[tuple[float, int, int]]] = {fid: [] for fid in feature_ids}

    captured: list[torch.Tensor] = []

    def hook_fn(module, inputs, output):  # noqa: ARG001
        captured.append(output[0] if isinstance(output, tuple) else output)

    layer = get_layer(model, preset, layer_idx)
    handle = layer.register_forward_hook(hook_fn)

    text = load_wikitext_train_text()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total = tokens.shape[0]
    n_batches = total // context_len
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    # Cache window token IDs per batch ID we keep in any heap; we'll decode
    # them at the end. Keyed by batch_id -> 1-D tensor of length context_len.
    kept_windows: dict[int, torch.Tensor] = {}

    print(f"[INFO] top_contexts: streaming {n_batches} windows of {context_len} "
          f"tokens for {len(feature_ids)} features (top_k={top_k})", flush=True)
    t0 = time.time()
    try:
        for bi in range(n_batches):
            window = tokens[bi * context_len:(bi + 1) * context_len]
            chunk = window.to(DEVICE)
            captured.clear()
            with torch.no_grad():
                _ = model(chunk.unsqueeze(0))
            resid = captured[0]                                 # [1, T, d_model]
            feats = sae.encode(resid)[0, :, feature_ids_t]      # [T, k]
            # For each feature, find max activation in this window.
            top_per_feat, top_pos = feats.max(dim=0)            # [k], [k]
            top_per_feat = top_per_feat.cpu().tolist()
            top_pos = top_pos.cpu().tolist()

            kept_window = False
            for fid in feature_ids:
                i = feat_pos[fid]
                act = top_per_feat[i]
                pos = top_pos[i]
                heap = heaps[fid]
                if len(heap) < top_k:
                    heapq.heappush(heap, (act, bi, pos))
                    kept_window = True
                elif act > heap[0][0]:
                    heapq.heapreplace(heap, (act, bi, pos))
                    kept_window = True
            if kept_window:
                kept_windows[bi] = window.clone()

            if (bi + 1) % log_every == 0:
                rate = (bi + 1) / max(time.time() - t0, 1e-6)
                eta = (n_batches - bi - 1) / rate
                print(f"  batch {bi+1}/{n_batches}  ({rate:.0f} b/s, eta {eta:.0f}s)"
                      f"  kept_windows={len(kept_windows)}", flush=True)
    finally:
        handle.remove()

    # Decode kept windows once.
    decoded: dict[int, list[str]] = {}
    for bi, w in kept_windows.items():
        decoded[bi] = tokenizer.convert_ids_to_tokens(w.tolist())

    out: dict = {}
    for fid in feature_ids:
        heap = sorted(heaps[fid], key=lambda x: -x[0])  # descending
        examples = []
        for act, bi, pos in heap:
            toks = decoded[bi]
            examples.append({
                "activation": float(act),
                "token_idx": int(pos),
                "tokens": toks,
                "activating_token": toks[pos] if 0 <= pos < len(toks) else "",
                "window_start_token": int(bi * context_len),
            })
        out[str(fid)] = {
            "max_activation": float(examples[0]["activation"]) if examples else 0.0,
            "examples": examples,
        }
    return out


def pretty_print(top_contexts: dict, max_examples_per_feature: int = 4) -> str:
    """Render top contexts as a readable text block."""

    def _show(t: str) -> str:
        return t.replace("Ġ", "·").replace("▁", "·").replace("\n", "↵")

    lines: list[str] = []
    for fid, payload in top_contexts.items():
        lines.append("=" * 78)
        lines.append(f"feature {fid}  max_activation={payload['max_activation']:.3f}")
        lines.append("=" * 78)
        for ex in payload["examples"][:max_examples_per_feature]:
            toks = ex["tokens"]
            pos = ex["token_idx"]
            cells = []
            for i, t in enumerate(toks):
                shown = _show(t)
                cells.append(f"[{shown}]" if i == pos else shown)
            lines.append(f"  act={ex['activation']:.3f}  "
                         f"@token={ex['activating_token']!r}")
            lines.append("    " + " ".join(cells))
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="gemma-2-2b")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--feature-ids", type=int, nargs="+", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--context-len", type=int, default=64)
    ap.add_argument("--max-batches", type=int, default=4000)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    out = collect_top_contexts(
        args.feature_ids, args.preset, args.layer,
        top_k=args.top_k, context_len=args.context_len,
        max_batches=args.max_batches,
    )

    out_dir = args.output_dir or (Path("data") / args.preset)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"top_contexts_layer{args.layer}.json"
    txt_path = out_dir / f"top_contexts_layer{args.layer}.txt"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(pretty_print(out))
    print(f"[INFO] wrote {json_path}")
    print(f"[INFO] wrote {txt_path}")


if __name__ == "__main__":
    main()
