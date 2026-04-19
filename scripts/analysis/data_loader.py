"""WikiText-2 corpus loader, on-the-fly from HuggingFace datasets.

All analysis scripts share this helper so the corpus is never materialised
to disk. The first call downloads + caches via the `datasets` library; subsequent
calls within the same process hit the in-memory cache.
"""

from __future__ import annotations

_cached_text: str | None = None


def load_wikitext_train_text() -> str:
    """Return the WikiText-2 raw train split as a single concatenated string."""
    global _cached_text
    if _cached_text is not None:
        return _cached_text
    from datasets import load_dataset
    print("[INFO] Loading WikiText-2 (wikitext-2-raw-v1) train split from HuggingFace...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    _cached_text = "".join(row["text"] for row in ds)
    return _cached_text
