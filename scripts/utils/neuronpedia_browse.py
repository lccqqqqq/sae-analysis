"""Sample a Neuronpedia SAE for candidate cherry-pick features.

Uniformly samples N feature indices from a public Neuronpedia SAE, fetches
their auto-interp explanation, top-activating contexts, activation density,
and (where present) the Paulo-et-al-style recall-detection score, then
classifies each into a tier (token / phrase / concept / abstract) using a
heuristic on the *top activations* — not the decoder→unembedding direct
attribution (`pos_str`), which is noisy at mid-layer because the residual
stream still has many transformer layers of computation left between the
SAE site and the unembedding.

The output is the *candidate list* — the user inspects it (and the
top-activating contexts via ``scripts/analysis/top_activating_contexts.py``)
before committing to a final ~20-25-feature pilot YAML.

Usage:
    python scripts/utils/neuronpedia_browse.py \\
        --model-id gemma-2-2b --sae-id 12-gemmascope-res-16k \\
        --n-latent 16384 --n-sample 300 \\
        --output data/gemma-2-2b/candidate_features_l12.md

Status note (2026-05-02): FROZEN. Half of this script's job — the
*tier classifier* (token / phrase / concept / abstract) — is no longer
trusted: the project has since concluded that tier labels carry no
signal beyond H_pos itself, and the heuristic was found to be of poor
accuracy on the candidate features. See
``feedback_use_h_pos_not_tier.md`` in auto-memory and the 2026-05-01
``CLEANUP_CHANGELOG.md`` entry for the surrounding context. The
*sampling + Neuronpedia-fetch + cache-population* half is still sound
and is what produced the 298-feature set used by the geometry
experiment (``data/feature_geometry_vs_entropy/``); that YAML is the
load-bearing artefact, this script is the procedure.

Companion script: ``scripts/utils/candidates_md_to_yaml.py``.
Together they are the "tier toolchain". If you want to extend the
feature set going forward, write a tier-free replacement — do not
treat this script's tier output as meaningful.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.neuronpedia_fetch import best_explanation, fetch_feature  # noqa: E402

# --- Tier classification heuristics -----------------------------------------

PHRASE_KEYWORDS = re.compile(
    r"\b(phrase|starting|ending|preceded|followed|punctuation|"
    r"after\b|before\b|between\b|n-?gram|bigram|trigram|construction)",
    re.IGNORECASE,
)
ABSTRACT_KEYWORDS = re.compile(
    r"\b(reasoning|causal|argument|narrative|sentiment|emotional|deception|"
    r"intent|contrast|comparison|negation|hypothes|reflect|long-range|"
    r"context)",
    re.IGNORECASE,
)
CONCEPT_KEYWORDS = re.compile(
    r"\b(related to|references to|topic|subject|domain|scientific|"
    r"medical|legal|biological|programming|code|mathematical|historical|"
    r"geographical|political|culinary|musical|astronomical)",
    re.IGNORECASE,
)


def _strip_token(t: str) -> str:
    return t.replace("▁", "").replace("Ġ", "").replace("\n", "").lower().strip()


def _peak_tokens(activations: list[dict], n: int = 3) -> list[str]:
    """Extract the activating (peak-value) token from the top n activations."""
    out = []
    for a in activations[:n]:
        if not isinstance(a, dict):
            continue
        toks = a.get("tokens") or []
        idx = a.get("maxValueTokenIndex")
        if idx is None or not (0 <= idx < len(toks)):
            continue
        out.append(toks[idx])
    return out


def _peak_bigrams(activations: list[dict], n: int = 3) -> list[tuple[str, str]]:
    """Top n (token-before-peak, peak) pairs."""
    out = []
    for a in activations[:n]:
        if not isinstance(a, dict):
            continue
        toks = a.get("tokens") or []
        idx = a.get("maxValueTokenIndex")
        if idx is None or not (1 <= idx < len(toks)):
            continue
        out.append((toks[idx - 1], toks[idx]))
    return out


def classify_tier(payload: dict) -> tuple[str, str]:
    """Tier from the top-activating tokens. Falls back to explanation keywords.

    Returns (tier, reason).
    """
    explanation = best_explanation(payload).lower()
    activations = payload.get("activations") or []
    peaks = _peak_tokens(activations, n=3)

    # Token-tier signal: top-3 peak tokens are the same word / morphological family.
    if len(peaks) >= 3:
        stripped = [_strip_token(t) for t in peaks]
        stripped = [s for s in stripped if s]
        if len(stripped) >= 3:
            if len(set(stripped)) == 1:
                return "token", f"all 3 top-activating tokens are {peaks[0]!r}"
            root = min(stripped, key=len)
            if len(root) >= 3 and all(
                root[: max(2, len(root) - 1)] in s or s in root for s in stripped
            ):
                return ("token",
                        f"top-3 activating tokens share a stem: {peaks}")

    # Phrase-tier signal: top-3 share the same preceding-token bigram.
    bigrams = _peak_bigrams(activations, n=3)
    if len(bigrams) >= 3:
        stripped_bg = [(_strip_token(a), _strip_token(b)) for a, b in bigrams]
        if len(set(stripped_bg)) == 1:
            return ("phrase",
                    f"top-3 share a bigram: {bigrams[0]}")
        # Soft signal: bigram structure not identical, but the explanation
        # itself mentions a phrase pattern.
        if PHRASE_KEYWORDS.search(explanation):
            return ("phrase",
                    "explanation mentions phrase/position; "
                    "peak bigrams not identical but explanation suggests it")

    # Abstract / concept fall-throughs.
    if ABSTRACT_KEYWORDS.search(explanation):
        return "abstract", "explanation contains abstract-reasoning keyword"
    if PHRASE_KEYWORDS.search(explanation):
        return "phrase", "explanation mentions phrase / position / punctuation"
    if CONCEPT_KEYWORDS.search(explanation):
        return "concept", "explanation references a topic / domain / semantic class"
    if len(explanation.split()) < 4:
        return "token", "very short explanation"
    return "concept", "default — non-trivial explanation, no other signal"


# --- Activation snippet rendering -------------------------------------------

def _show(t: str) -> str:
    return t.replace("▁", "·").replace("Ġ", "·").replace("\n", "↵")


def render_activation_snippet(activation: dict, k_before: int = 4,
                              k_after: int = 3) -> str:
    """Window of ±k tokens around the peak; peak token marked with [...]."""
    if not isinstance(activation, dict):
        return ""
    toks = activation.get("tokens") or []
    idx = activation.get("maxValueTokenIndex")
    val = activation.get("maxValue")
    if idx is None or not (0 <= idx < len(toks)):
        return ""
    lo = max(0, idx - k_before)
    hi = min(len(toks), idx + k_after + 1)
    cells = []
    for i in range(lo, hi):
        s = _show(toks[i])
        cells.append(f"**[{s}]**" if i == idx else s)
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(toks) else ""
    return f"{prefix}{' '.join(cells)}{suffix}  *(peak={val:.2f})*"


def render_top_activations(payload: dict, n: int = 3) -> str:
    activations = payload.get("activations") or []
    snippets = []
    for a in activations[:n]:
        s = render_activation_snippet(a)
        if s:
            snippets.append(s)
    return "<br>".join(snippets)


def get_recall_score(payload: dict) -> float | None:
    """Highest recall_alt score across the feature's explanations, or None."""
    best = None
    for e in payload.get("explanations") or []:
        for s in e.get("scores") or []:
            if (isinstance(s, dict)
                    and s.get("explanationScoreTypeName") == "recall_alt"):
                v = s.get("value")
                if isinstance(v, (int, float)):
                    best = v if best is None else max(best, v)
    return best


# --- Sampling driver --------------------------------------------------------

def fetch_one(args: tuple[str, str, int]) -> dict | None:
    model_id, sae_id, idx = args
    try:
        return fetch_feature(model_id, sae_id, idx)
    except Exception as e:
        return {"_error": str(e), "_index": idx}


def run(model_id: str, sae_id: str, n_latent: int, n_sample: int,
        seed: int, max_workers: int, density_lo: float, density_hi: float,
        output_path: Path):
    rng = random.Random(seed)
    indices = rng.sample(range(n_latent), k=min(n_sample, n_latent))
    print(f"[INFO] sampling {len(indices)} features from {model_id}/{sae_id}",
          flush=True)

    payloads: list[dict] = []
    work = [(model_id, sae_id, i) for i in indices]
    n_done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, w): w[2] for w in work}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None and "_error" not in r:
                payloads.append(r)
            n_done += 1
            if n_done % 50 == 0:
                print(f"  fetched {n_done}/{len(indices)}", flush=True)
    print(f"[INFO] fetched {len(payloads)} valid payloads "
          f"({len(indices) - len(payloads)} errors)", flush=True)

    rows = []
    for p in payloads:
        density = p.get("frac_nonzero")
        if density is None or density <= 0:
            continue
        if not (density_lo <= density <= density_hi):
            continue
        expl = best_explanation(p)
        if not expl or expl == "(no explanation)":
            continue
        tier, reason = classify_tier(p)
        rows.append({
            "feature_id": int(p["index"]),
            "tier": tier,
            "tier_reason": reason,
            "explanation": expl,
            "density": density,
            "max_act": p.get("maxActApprox"),
            "recall": get_recall_score(p),
            "activations_md": render_top_activations(p, n=3),
        })

    rows.sort(key=lambda r: (
        ["token", "phrase", "concept", "abstract"].index(r["tier"]),
        -r["density"],
    ))

    write_markdown(rows, model_id, sae_id, n_latent, n_sample, len(payloads),
                   output_path)
    n_scored = sum(1 for r in rows if r["recall"] is not None)
    print(f"[INFO] wrote {output_path}  "
          f"({len(rows)} candidates, {n_scored} with recall_alt score)",
          flush=True)


def write_markdown(rows: list[dict], model_id: str, sae_id: str, n_latent: int,
                   n_sample: int, n_fetched: int, path: Path):
    by_tier: dict[str, list[dict]] = {"token": [], "phrase": [],
                                      "concept": [], "abstract": []}
    for r in rows:
        by_tier[r["tier"]].append(r)
    n_scored = sum(1 for r in rows if r["recall"] is not None)

    lines = [
        f"# Candidate cherry-pick features — {model_id} / {sae_id}",
        "",
        f"- SAE width: {n_latent} features",
        f"- Random sample: {n_sample} (seed-stable)",
        f"- Successfully fetched: {n_fetched}",
        f"- After filtering by density and presence of auto-interp: **{len(rows)}**",
        f"- Density filter: only features with `frac_nonzero` in [1e-5, 1e-1]",
        f"- Recall-detection score (Paulo et al. 2024 method, judge=gpt-4o-mini) "
        f"available for **{n_scored} of {len(rows)}** candidates.",
        "",
        "Each row shows the **top 3 activating contexts** with the peak token "
        "marked **[token]**, decoded with `·` for leading-space and `↵` for "
        "newline. The tier label is a heuristic on those peaks (not on the "
        "decoder-to-unembedding logit attribution, which is noisy at mid-layer "
        "and is what produced the weird `pos_str` in the previous version).",
        "",
        "Verify by reading the activations: if the peak token / surrounding "
        "context is consistent across all three examples and matches the "
        "explanation, the feature is a good cherry-pick. If the contexts look "
        "unrelated, the feature is polysemantic and should be skipped.",
        "",
    ]
    for tier in ["token", "phrase", "concept", "abstract"]:
        lines.append(f"## Tier: {tier}  ({len(by_tier[tier])} candidates)")
        lines.append("")
        if not by_tier[tier]:
            lines.append("_(none in this sample)_")
            lines.append("")
            continue
        lines.append("| feature_id | density | recall | explanation | "
                     "top-3 activations (peak token in **[...]**) |")
        lines.append("|---:|---:|---:|---|---|")
        for r in by_tier[tier]:
            density = f"{r['density']:.2e}"
            recall = f"{r['recall']:.2f}" if r["recall"] is not None else "—"
            expl = r["explanation"].replace("|", "\\|").strip()
            if len(expl) > 90:
                expl = expl[:87] + "…"
            acts = r["activations_md"].replace("|", "\\|")
            lines.append(
                f"| {r['feature_id']} | {density} | {recall} | "
                f"{expl} | {acts} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="gemma-2-2b")
    ap.add_argument("--sae-id", default="12-gemmascope-res-16k")
    ap.add_argument("--n-latent", type=int, default=16384)
    ap.add_argument("--n-sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--density-lo", type=float, default=1e-5)
    ap.add_argument("--density-hi", type=float, default=1e-1)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if args.output is None:
        layer = args.sae_id.split("-")[0]
        args.output = (Path("data") / args.model_id /
                       f"candidate_features_l{layer}.md")

    run(args.model_id, args.sae_id, args.n_latent, args.n_sample,
        args.seed, args.max_workers, args.density_lo, args.density_hi,
        args.output)


if __name__ == "__main__":
    main()
