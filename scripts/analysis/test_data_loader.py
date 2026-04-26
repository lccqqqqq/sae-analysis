"""Tests for ``scripts/analysis/data_loader.py``.

Verifies the corpus loader returns a non-empty string, that the WikiText-2
*raw* variant is in fact what's loaded (no ``<unk>`` / ``@-@`` artefacts),
that the in-process cache returns identity-equal objects, and that the
generic ``load_hf_text`` entry point shares the same cache as the
WikiText-specific wrapper.

Run from repo root or anywhere:

    python scripts/analysis/test_data_loader.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly: put the analysis directory on sys.path
# so sibling import of ``data_loader`` works regardless of cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_loader import load_hf_text, load_wikitext_train_text


def test_loads_nonempty_string():
    text = load_wikitext_train_text()
    assert isinstance(text, str), f"expected str, got {type(text).__name__}"
    # WikiText-2 train is ~10MB raw; a sane lower bound is 1MB.
    assert len(text) > 1_000_000, (
        f"WikiText-2 train should be >1MB; got {len(text)} chars."
    )


def test_no_unk_contamination():
    """The raw variant must not contain ``<unk>`` substitution.

    Note: wikitext-2-raw-v1 *does* still carry the Merity-style ``@-@``
    hyphenation and space-separated punctuation -- those artefacts are
    inherent to both WikiText-2 distributions on HuggingFace and are not
    fixable by switching configs. Only ``<unk>`` and case-lowercasing
    distinguish raw-v1 from v1, so we assert only the ``<unk>``-free
    invariant here.
    """
    text = load_wikitext_train_text()
    assert "<unk>" not in text, (
        "WikiText-2 raw variant unexpectedly contains <unk>; "
        "the preprocessed variant (wikitext-2-v1) was loaded instead of "
        "wikitext-2-raw-v1."
    )


def test_case_preserved():
    """Raw variant preserves case (preprocessed variant is fully lowercased)."""
    text = load_wikitext_train_text()
    # The first article is "Valkyria Chronicles III"; the proper noun should
    # appear with capitalisation. In the lowercased preprocessed variant,
    # this exact capitalisation would not be present.
    assert "Valkyria" in text, (
        "Expected mixed-case proper nouns to be preserved in raw-v1; "
        "got an all-lowercase corpus -- did the preprocessed variant load?"
    )


def test_in_process_cache_returns_same_object():
    a = load_wikitext_train_text()
    b = load_wikitext_train_text()
    assert a is b, "Repeated calls should return the cached string object."


def test_generic_loader_shares_cache():
    """``load_hf_text`` and the wrapper hit the same cache key."""
    a = load_wikitext_train_text()
    b = load_hf_text("wikitext", "wikitext-2-raw-v1", "train")
    assert a is b, (
        "load_hf_text and load_wikitext_train_text should resolve to the "
        "same cached object."
    )


def test_different_split_is_distinct():
    """Different (dataset, config, split) keys must NOT collide in the cache."""
    train = load_wikitext_train_text()
    test = load_hf_text("wikitext", "wikitext-2-raw-v1", "test")
    assert train is not test, "train and test splits should be distinct objects."
    assert "<unk>" not in test, (
        "WikiText-2 raw test split should also be unk-free."
    )


if __name__ == "__main__":
    print("Running data_loader tests...")
    for fn in [
        test_loads_nonempty_string,
        test_no_unk_contamination,
        test_case_preserved,
        test_in_process_cache_returns_same_object,
        test_generic_loader_shares_cache,
        test_different_split_is_distinct,
    ]:
        fn()
        print(f"  ok  {fn.__name__}")
    print("All tests passed.")
