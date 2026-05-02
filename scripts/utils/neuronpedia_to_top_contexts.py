"""Adapter: cached Neuronpedia activations -> virtual-corpus + top_contexts JSON.

Reuses the same on-disk format that ``scripts/analysis/top_activating_contexts.py``
emits, so ``scripts/analysis/topact_entropy.py`` can be pointed at the
output with no logic change beyond a ``--corpus-tensor`` flag.

Inputs:
  - YAML with the 20 cherry-picked feature ids.
  - Cached Neuronpedia payloads under
      data/neuronpedia_cache/<model_id>/<sae_id>/<fid>.json
    (already populated from earlier browse-step API calls).

Outputs (under ``--output-dir``):
  - ``corpus_tokens.pt``       1-D LongTensor of all retained events' token
                               IDs, with a ``<bos>`` prepended per event so
                               the local forward pass matches Neuronpedia's
                               ``prepend_bos: true`` convention.
  - ``top_contexts_layer<L>.json``  feature-id -> {"max_activation", "examples":
                               [{activation, token_idx, tokens, activating_token,
                                 window_start_token, neuronpedia_data_index,
                                 neuronpedia_rank}]}.
  - ``adapter_log.json``       summary: per-feature event counts before/after
                               dedup + vocab-roundtrip filtering.

Dedup criterion: ``hash(tuple(tokens), maxValueTokenIndex)``. Same-text
event with same peak position is considered identical regardless of
``dataIndex`` (pile sometimes contains near-duplicate documents).

Vocab round-trip: each retained event must satisfy
``convert_ids_to_tokens(convert_tokens_to_ids(toks)) == toks``. Events
that fail are dropped with a per-token reason logged. Failures should be
rare for Gemma's SentencePiece vocab; persistent failures usually indicate
a special token that needs custom handling.

Activation match against Neuronpedia's ``maxValue`` is *not* done here
(needs GPU). It is performed by ``scripts/analysis/verify_neuronpedia_match.py``
as a separate small smoke job before the parallel entropy jobs are
submitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "analysis"))

from presets import get_preset  # noqa: E402

DEFAULT_CACHE_ROOT = Path("data") / "neuronpedia_cache"


def _vocab_roundtrip_ok(tokenizer, tokens: list[str]) -> tuple[bool, str]:
    """Strict vocab identity check; returns (ok, reason)."""
    try:
        ids = tokenizer.convert_tokens_to_ids(tokens)
    except Exception as e:
        return False, f"convert_tokens_to_ids raised {e!r}"
    if any(i is None for i in ids):
        return False, "some tokens mapped to None"
    rt = tokenizer.convert_ids_to_tokens(ids)
    if rt != tokens:
        for i, (a, b) in enumerate(zip(tokens, rt)):
            if a != b:
                return False, f"mismatch at index {i}: {a!r} vs {b!r}"
        return False, f"length mismatch: {len(tokens)} vs {len(rt)}"
    return True, ""


def run(
    yaml_path: Path,
    output_dir: Path,
    cache_root: Path,
    top_k: int,
):
    cfg = yaml.safe_load(yaml_path.read_text())
    preset = get_preset(cfg["preset"])
    layer_idx = int(cfg["layer"])
    sae_id = str(cfg["sae_id"])
    feats = cfg["features"]

    # CPU-only: tokenizer load is fast; no model needed for the adapter.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(preset.model_id)
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        # Gemma-2's tokenizer always has a BOS; this would be a config issue.
        raise RuntimeError(
            f"tokenizer for {preset.model_id} has no bos_token_id; "
            "neuronpedia uses prepend_bos:true and we need it to match.")
    print(f"[INFO] tokenizer={preset.model_id}  bos_id={bos_id}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_chunks: list[torch.Tensor] = []
    json_blob: dict[str, dict] = {}
    log: dict[str, dict] = {}

    cur_offset = 0
    for entry in feats:
        fid = int(entry["feature_id"])
        cache_path = cache_root / preset.name / sae_id / f"{fid}.json"
        if not cache_path.exists():
            print(f"[WARN] missing cache for feature {fid}: {cache_path}",
                  flush=True)
            continue
        payload = json.loads(cache_path.read_text())
        activations = payload.get("activations") or []

        # Dedup: hash (tokens-tuple, peak-index).
        seen: set[tuple] = set()
        dedup: list[dict] = []
        n_dup = 0
        for a in activations:
            if not isinstance(a, dict):
                continue
            toks = a.get("tokens") or []
            peak = a.get("maxValueTokenIndex")
            if peak is None or not (0 <= peak < len(toks)):
                continue
            key = (tuple(toks), int(peak))
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            dedup.append(a)
        # Sort by maxValue desc (Neuronpedia usually already sorted, but be safe).
        dedup.sort(key=lambda a: -float(a.get("maxValue", 0.0)))

        # Vocab round-trip filter.
        retained = []
        n_vocab_drop = 0
        roundtrip_failures: list[str] = []
        for a in dedup:
            toks = a["tokens"]
            ok, reason = _vocab_roundtrip_ok(tokenizer, toks)
            if not ok:
                n_vocab_drop += 1
                roundtrip_failures.append(reason)
                continue
            retained.append(a)
            if len(retained) >= top_k:
                break

        examples_out = []
        for rank, a in enumerate(retained):
            toks = a["tokens"]
            peak = int(a["maxValueTokenIndex"])
            ids = tokenizer.convert_tokens_to_ids(toks)
            # Prepend BOS to match neuronpedia's prepend_bos:true.
            event_ids = torch.tensor([bos_id] + list(ids), dtype=torch.long)
            corpus_chunks.append(event_ids)
            # New peak index inside this event's slice (BOS shifts by 1).
            new_peak = peak + 1
            examples_out.append({
                "activation": float(a.get("maxValue", 0.0)),
                "token_idx": new_peak,
                "tokens": ["<bos>"] + list(toks),  # human-readable
                "activating_token": toks[peak] if 0 <= peak < len(toks) else "",
                "window_start_token": cur_offset,
                "neuronpedia_data_index": a.get("dataIndex"),
                "neuronpedia_rank": rank,
            })
            cur_offset += event_ids.numel()

        json_blob[str(fid)] = {
            "max_activation": float(retained[0]["maxValue"]) if retained else 0.0,
            "examples": examples_out,
        }
        log[str(fid)] = {
            "tier": entry["tier"],
            "n_neuronpedia": len(activations),
            "n_after_dedup": len(dedup),
            "n_duplicates": n_dup,
            "n_vocab_dropped": n_vocab_drop,
            "n_kept": len(retained),
            "roundtrip_failure_examples": roundtrip_failures[:3],
        }
        print(f"[INFO] feature {fid:>5d} ({entry['tier']:<8s})  "
              f"raw={len(activations):>2d}  dedup_drop={n_dup:>2d}  "
              f"vocab_drop={n_vocab_drop:>2d}  kept={len(retained):>2d}",
              flush=True)

    # Materialise the virtual corpus tensor.
    if not corpus_chunks:
        raise RuntimeError("no events retained; nothing to save")
    corpus_tokens = torch.cat(corpus_chunks)

    corpus_path = output_dir / "corpus_tokens.pt"
    json_path = output_dir / f"top_contexts_layer{layer_idx}.json"
    log_path = output_dir / "adapter_log.json"
    torch.save(corpus_tokens, corpus_path)
    json_path.write_text(json.dumps(json_blob, indent=2))
    log_path.write_text(json.dumps(log, indent=2))

    total_events = sum(len(v["examples"]) for v in json_blob.values())
    print(f"\n[INFO] virtual corpus: {corpus_tokens.numel():,} tokens",
          flush=True)
    print(f"[INFO] retained {total_events} events across "
          f"{len(json_blob)} features", flush=True)
    print(f"[INFO] wrote {corpus_path}")
    print(f"[INFO] wrote {json_path}")
    print(f"[INFO] wrote {log_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--top-k", type=int, default=32)
    args = ap.parse_args()
    run(args.yaml, args.output_dir, args.cache_root, args.top_k)


if __name__ == "__main__":
    main()
