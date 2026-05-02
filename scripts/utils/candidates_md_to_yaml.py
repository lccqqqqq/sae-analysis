"""Convert the candidate-features markdown into a pilot-style YAML.

Reads ``data/<model>/candidate_features_l<L>.md`` (the random-sample
output from ``scripts/utils/neuronpedia_browse.py``) and emits a YAML
in the same shape as
``scripts/experiments/configs/pilot_features_gemma2_2b_l12.yaml`` so
the existing entropy pipeline can consume it unchanged.

Tier of each feature is taken from the ``## Tier: <name>`` headers in
the markdown (the heuristic classification done at browse time).
Description is the auto-interp explanation column from the markdown.

Usage:
    python scripts/utils/candidates_md_to_yaml.py \\
        --input data/gemma-2-2b/candidate_features_l12.md \\
        --output scripts/experiments/configs/full_features_gemma2_2b_l12.yaml \\
        --preset gemma-2-2b --layer 12 --sae-id 12-gemmascope-res-16k

Status note (2026-05-02): FROZEN. End-to-end built around the dead
tier scheme (token / phrase / concept / abstract): expects
``## Tier: <name>`` headers in the input markdown and writes a
``tier:`` field per feature into the output YAML. Per
``feedback_use_h_pos_not_tier.md`` in auto-memory, tier labels carry
no signal beyond H_pos itself; they survive in the live pipeline only
as a passthrough group key for ``topact_entropy.py --tier-filter``.
The 298-feature YAML this script produced is the load-bearing
artefact — this script is the procedure that generated it. Use it
only to regenerate that YAML; for extending the feature set, write a
tier-free replacement.

Companion script: ``scripts/utils/neuronpedia_browse.py``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TIER_HEADER = re.compile(r"^##\s+Tier:\s*(\w+)\s*\(\d+\s+candidates\)")
ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([\deE.+-]+)\s*\|\s*([^|]+?)\s*\|"
                 r"\s*([^|]+?)\s*\|")


def parse(md_path: Path) -> list[dict]:
    features: list[dict] = []
    current_tier = None
    for line in md_path.read_text().splitlines():
        h = TIER_HEADER.match(line)
        if h:
            current_tier = h.group(1)
            continue
        if current_tier is None:
            continue
        m = ROW.match(line)
        if not m:
            continue
        fid = int(m.group(1))
        desc = m.group(4).strip().rstrip("…").strip().rstrip(".").strip()
        # Strip any backslash escapes for | inside the description.
        desc = desc.replace("\\|", "|")
        features.append({
            "feature_id": fid,
            "tier": current_tier,
            "description": desc,
        })
    return features


def write_yaml(features: list[dict], preset: str, layer: int, sae_id: str,
               output_path: Path):
    lines = [
        f"# Auto-generated from candidates markdown by "
        f"scripts/utils/candidates_md_to_yaml.py.",
        f"# Tier classifications come from the heuristic in "
        f"scripts/utils/neuronpedia_browse.py.",
        "",
        f"preset: {preset}",
        f"layer: {layer}",
        f"sae_id: {sae_id}",
        "",
        "features:",
    ]
    # Group features by tier to match the human-edited yaml's structure.
    for tier in ("token", "phrase", "concept", "abstract"):
        sub = [f for f in features if f["tier"] == tier]
        if not sub:
            continue
        lines.append(f"  # ----- {tier} tier ({len(sub)}) -----")
        for f in sub:
            desc = f["description"].replace('"', '\\"')
            lines.append(
                f"  - {{feature_id: {f['feature_id']:>5d}, tier: {tier:<8s}, "
                f'description: "{desc}"}}'
            )
        lines.append("")
    output_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--preset", type=str, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sae-id", type=str, required=True)
    args = ap.parse_args()
    features = parse(args.input)
    counts: dict[str, int] = {}
    for f in features:
        counts[f["tier"]] = counts.get(f["tier"], 0) + 1
    print(f"[INFO] parsed {len(features)} features: " +
          ", ".join(f"{t}={n}" for t, n in counts.items()))
    write_yaml(features, args.preset, args.layer, args.sae_id, args.output)
    print(f"[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()
