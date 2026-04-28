"""Text corpus loaders for the SAE analysis pipeline.

The analysis scripts under ``scripts/analysis/`` previously read a local
``wikitext-2-train.txt`` checked into the repo. That file is the
*preprocessed* variant of WikiText-2 (Merity et al. 2016), which has two
relevant departures from natural Wikipedia text:

1. Rare words are replaced with the literal string ``<unk>`` (14,334
   occurrences in the train split, ~0.7% of all tokens). Pythia's BPE
   tokenizer does *not* treat ``<unk>`` as a special unknown-token
   marker -- it gets BPE-segmented into raw characters (``<``, ``un``,
   ``k``, ``>``) that never occur in Pythia's pretraining distribution.
2. Hyphens / decimal points / commas in numbers are encoded as ``@-@`` /
   ``@.@`` / ``@,@``, and punctuation is space-separated. Less harmful
   than ``<unk>`` (the ``@`` glyph does occur in natural text) but still
   off-distribution.

Switching to the ``wikitext-2-raw-v1`` HuggingFace config fixes (1) and
also restores the original case of proper nouns. It does **not** fix (2):
the ``@-@`` artefacts are inherent to both WikiText-2 distributions on
HuggingFace. A future port to a different corpus (e.g. ``wikipedia`` or
``the_pile``) would be required to remove them. See ``CHANGELOG.md`` at
the repo root for the full impact analysis.

This module provides:

- ``load_wikitext_train_text()`` -- drop-in default. Returns the
  WikiText-2 *raw* train split as a single concatenated string.
- ``load_hf_text(dataset, config, split)`` -- general entry point for any
  HuggingFace text dataset whose rows expose a ``"text"`` field.

Both go through the same in-process cache so a single Python process
never re-downloads or re-concatenates the same corpus.
"""

from __future__ import annotations


# Process-level cache keyed by (dataset, config, split). Identity-stable:
# repeated calls with the same key return the same string object.
_text_cache: dict[tuple[str, str, str], str] = {}


def load_hf_text(dataset: str, config: str, split: str = "train") -> str:
    """Return the concatenated ``text`` field of a HuggingFace text dataset.

    Args:
        dataset: HF dataset name (e.g. ``"wikitext"``).
        config:  HF dataset config / subset (e.g. ``"wikitext-2-raw-v1"``).
        split:   ``"train"`` / ``"validation"`` / ``"test"``.

    Returns:
        The concatenation of every row's ``"text"`` field, in dataset order.
    """
    key = (dataset, config, split)
    if key in _text_cache:
        return _text_cache[key]
    from datasets import load_dataset
    print(
        f"[INFO] Loading {dataset}/{config}[{split}] from HuggingFace...",
        flush=True,
    )
    ds = load_dataset(dataset, config, split=split)
    text = "".join(row["text"] for row in ds)
    _text_cache[key] = text
    return text


def load_wikitext_train_text() -> str:
    """Return the WikiText-2 raw-v1 train split as a single string.

    This is the default corpus for the analysis pipeline. Use the *raw*
    variant to avoid the ``<unk>`` / ``@-@`` contamination present in the
    preprocessed ``wikitext-2-v1`` distribution.
    """
    return load_hf_text("wikitext", "wikitext-2-raw-v1", "train")
