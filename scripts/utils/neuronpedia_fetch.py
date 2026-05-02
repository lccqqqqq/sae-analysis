"""HTTP client for Neuronpedia's public feature API.

Neuronpedia hosts auto-interp dashboards for many open SAEs (Gemma Scope,
sae_lens releases, etc.). The single-feature endpoint is

    https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{index}

It returns JSON with `pos_str`, `neg_str`, `frac_nonzero`, `explanations`
(list of auto-interp objects, each with a `description` string and zero or
more evaluation scores), and `activations` (top activating examples). No
authentication is required for public SAEs.

Responses are cached on disk under ``data/neuronpedia_cache/`` so repeated
calls are cheap and reproducible. To bust the cache for one feature, delete
its file under that directory.

This module deliberately contains no analysis; it is engineering plumbing.
Analysis-side files in ``scripts/analysis/`` should import ``fetch_feature``
and read from the returned dict.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

API_TEMPLATE = "https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{index}"
DEFAULT_CACHE_DIR = Path("data") / "neuronpedia_cache"
USER_AGENT = "sae-analysis/research-pilot (github.com/lccqqqqq)"


def _cache_path(cache_dir: Path, model_id: str, sae_id: str, index: int) -> Path:
    safe_sae = sae_id.replace("/", "__")
    return cache_dir / model_id / safe_sae / f"{index}.json"


def fetch_feature(
    model_id: str,
    sae_id: str,
    index: int,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    timeout: float = 30.0,
    refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.5,
) -> dict:
    """Return the Neuronpedia feature payload as a dict (cached on disk).

    Args:
        model_id: e.g. ``"gemma-2-2b"``.
        sae_id:   e.g. ``"12-gemmascope-res-16k"`` (layer 12, residual, 16k width).
        index:    feature index in the SAE.
        cache_dir: root for on-disk JSON cache.
        timeout:  per-request timeout in seconds.
        refresh:  if True, ignore the cache and re-fetch.
        retries:  total request attempts on transient failure.
        backoff:  exponential factor applied to ``time.sleep`` between retries.

    Raises:
        urllib.error.HTTPError on non-2xx responses; ValueError if JSON is
        malformed.
    """
    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, model_id, sae_id, index)

    if not refresh and path.exists():
        with open(path) as f:
            return json.load(f)

    url = API_TEMPLATE.format(model_id=model_id, sae_id=sae_id, index=index)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
            else:
                raise

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    return payload


def best_explanation(payload: dict) -> str:
    """Return a short human-readable description for the feature.

    Picks the first auto-interp explanation if present, falls back to the
    top three boosted output tokens. Used for plot annotations and
    sanity-check printouts.
    """
    expls = payload.get("explanations") or []
    if expls and isinstance(expls, list):
        first = expls[0]
        if isinstance(first, dict) and first.get("description"):
            return first["description"]
    pos = payload.get("pos_str") or []
    if pos:
        return "boosts: " + ", ".join(repr(t) for t in pos[:3])
    return "(no explanation)"


def gemma_scope_sae_id(layer: int, width_k: int = 16) -> str:
    """Convenience: Neuronpedia ``sae_id`` for Gemma-2-2B residual-stream SAEs."""
    return f"{layer}-gemmascope-res-{width_k}k"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="gemma-2-2b")
    ap.add_argument("--sae-id", default="12-gemmascope-res-16k")
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    payload = fetch_feature(
        args.model_id, args.sae_id, args.index, refresh=args.refresh,
    )
    print(f"index   : {payload['index']}")
    print(f"density : {payload.get('frac_nonzero')}")
    print(f"pos_str : {payload.get('pos_str', [])[:5]}")
    print(f"explain : {best_explanation(payload)}")
