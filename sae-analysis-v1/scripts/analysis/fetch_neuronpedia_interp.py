"""Fetch Neuronpedia autointerp explanations for the top-5-colored features
in each batch of the entropy_comparison analysis.

The "colored" features are the top-5 by last-token activation per batch per
layer — the same set rendered in non-grey in scripts/plotting/regenerate_batch_plots.py.

Outputs
-------
data/neuronpedia_explanations.json
    {
      "layer_<N>": {
        "<feat_idx>": {
          "explanations": [ {"description", "model", "type", "author", "created_at"}, ... ],
          "max_act_approx": float,
          "pos_str": [...], "pos_values": [...],     # top promoted logits
          "neg_str": [...], "neg_values": [...],     # top suppressed logits
        },
        ...
      },
      ...
    }

data/neuronpedia_explanations.csv
    layer, feat_idx, description, explanation_model, type, max_act, top_pos_tokens, top_neg_tokens

Usage
-----
    # Loads NEURONPEDIA_API_KEY from .env in repo root (or env var).
    python3 scripts/analysis/fetch_neuronpedia_interp.py

    # Subset of layers:
    python3 scripts/analysis/fetch_neuronpedia_interp.py --layers 3 5
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import requests
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://www.neuronpedia.org/api"
MODEL_ID = "pythia-70m-deduped"
SAE_NAME = "res-sm"   # saprmarks residual-stream SAEs on Neuronpedia


def load_api_key():
    key = os.environ.get("NEURONPEDIA_API_KEY")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("NEURONPEDIA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    print("[ERROR] NEURONPEDIA_API_KEY not found in env or .env")
    sys.exit(1)


def collect_colored_features(data_dir: Path, layers):
    """For each layer, collect the set of unique feature indices that appear
    in the top-5-by-activation of any batch."""
    out = defaultdict(set)
    for layer in layers:
        files = list(data_dir.glob(f"entropy_comparison_resid_out_layer{layer}_*.pt"))
        if not files:
            print(f"[WARN] No data file for layer {layer}; skipping.")
            continue
        latest = max(files, key=lambda p: p.stat().st_mtime)
        data = torch.load(latest, map_location="cpu", weights_only=False)
        for br in data["batch_results"]:
            acts = br.get("feature_activations", {})
            if not acts:
                continue
            top5 = sorted(acts.items(), key=lambda x: x[1], reverse=True)[:5]
            for fid, _ in top5:
                out[layer].add(int(fid))
    return {l: sorted(out[l]) for l in sorted(out)}


def fetch_feature(session, layer, feat_idx, api_key, max_retries=3):
    """Fetch the feature record; return parsed JSON or None on permanent failure."""
    sae_id = f"{layer}-{SAE_NAME}"
    url = f"{BASE}/feature/{MODEL_ID}/{sae_id}/{feat_idx}"
    headers = {"X-Api-Key": api_key}
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:   # rate-limited
                wait = 2 ** attempt
                print(f"  [rate-limited] sleeping {wait}s...")
                time.sleep(wait)
                continue
            print(f"  [HTTP {r.status_code}] layer={layer} feat={feat_idx}")
            return None
        except requests.RequestException as e:
            print(f"  [ERROR] {e}; retry {attempt+1}/{max_retries}")
            time.sleep(1 + attempt)
    return None


def summarize(record):
    """Extract the fields we care about from the full API response."""
    if not record:
        return None
    expls = []
    for e in record.get("explanations") or []:
        expls.append({
            "description": e.get("description"),
            "model": e.get("explanationModelName"),
            "type": e.get("typeName"),
            "author": (e.get("author") or {}).get("name") if isinstance(e.get("author"), dict) else e.get("author"),
            "created_at": e.get("createdAt"),
        })
    return {
        "explanations": expls,
        "max_act_approx": record.get("maxActApprox"),
        "pos_str": record.get("pos_str"),
        "pos_values": record.get("pos_values"),
        "neg_str": record.get("neg_str"),
        "neg_values": record.get("neg_values"),
        "frac_nonzero": record.get("frac_nonzero"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-json", type=Path,
                        default=ROOT / "data" / "neuronpedia_explanations.json")
    parser.add_argument("--out-csv", type=Path,
                        default=ROOT / "data" / "neuronpedia_explanations.csv")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Seconds between requests (default 0.3)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip features already present in the output JSON.")
    args = parser.parse_args()

    api_key = load_api_key()
    print(f"[INFO] Using API key ending ...{api_key[-4:]}")

    feat_map = collect_colored_features(args.data_dir, args.layers)
    total = sum(len(v) for v in feat_map.values())
    print(f"[INFO] Colored-feature counts: {[f'L{l}:{len(v)}' for l, v in feat_map.items()]}")
    print(f"[INFO] Total unique (layer, feat) to query: {total}")

    # Resume support
    out = {}
    if args.resume and args.out_json.exists():
        try:
            out = json.loads(args.out_json.read_text())
            already = sum(len(v) for v in out.values())
            print(f"[INFO] Resuming; {already} features already in {args.out_json.name}")
        except Exception as e:
            print(f"[WARN] Could not parse existing JSON, starting fresh: {e}")
            out = {}

    session = requests.Session()
    i = 0
    start = time.time()
    for layer, feats in feat_map.items():
        key_layer = f"layer_{layer}"
        out.setdefault(key_layer, {})
        for fid in feats:
            i += 1
            if str(fid) in out[key_layer]:
                continue  # already have it
            record = fetch_feature(session, layer, fid, api_key)
            summary = summarize(record)
            if summary is not None:
                out[key_layer][str(fid)] = summary
            if i % 25 == 0:
                elapsed = time.time() - start
                rate = i / max(elapsed, 0.01)
                eta = (total - i) / max(rate, 0.01)
                print(f"  Progress: {i}/{total} | {rate:.1f} req/s | ETA ~{eta:.0f}s")
                # checkpoint
                args.out_json.parent.mkdir(parents=True, exist_ok=True)
                args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            time.sleep(args.sleep)

    # Final save
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[INFO] Wrote {args.out_json}")

    # CSV view (one row per explanation; features with no explanation get an empty row)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "feat_idx", "description", "explanation_model",
                    "type", "max_act_approx", "frac_nonzero",
                    "top_pos_tokens", "top_neg_tokens"])
        for key_layer, feats in out.items():
            layer = int(key_layer.split("_")[1])
            for fid, s in sorted(feats.items(), key=lambda x: int(x[0])):
                pos_tokens = ", ".join((s.get("pos_str") or [])[:5])
                neg_tokens = ", ".join((s.get("neg_str") or [])[:5])
                expls = s.get("explanations") or [{}]
                for e in expls:
                    w.writerow([layer, fid, e.get("description") or "",
                                e.get("model") or "", e.get("type") or "",
                                s.get("max_act_approx"), s.get("frac_nonzero"),
                                pos_tokens, neg_tokens])
    print(f"[INFO] Wrote {args.out_csv}")

    # Summary
    n_with_expl = 0
    n_total = 0
    for feats in out.values():
        for s in feats.values():
            n_total += 1
            if s.get("explanations"):
                n_with_expl += 1
    print(f"[INFO] {n_with_expl}/{n_total} features have at least one explanation")


if __name__ == "__main__":
    main()
