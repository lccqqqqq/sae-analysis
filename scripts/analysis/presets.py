"""
Model + SAE preset registry.

A preset bundles the HuggingFace model id, the SAE source (dictionary_learning
checkpoint path or sae_lens release), the SAE architecture (relu / jumprelu /
topk), the model-attribute paths needed to swap in a DummyEmbed + hook a layer,
and the analysis defaults (threshold, site naming).

To add a new (model, SAE) pair: add one entry to PRESETS. No other code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Preset:
    name: str
    # Model
    model_id: str
    num_layers: int
    d_model: int
    embed_path: str           # dotted path, e.g. "gpt_neox.embed_in"
    layer_path: str           # dotted path to ModuleList, e.g. "gpt_neox.layers"
    # SAE source
    sae_loader: str           # "dictionary_learning" | "sae_lens" | "sparsify"
    sae_arch: str             # "relu" | "jumprelu" | "topk"
    sae_path_template: Optional[str] = None   # for dictionary_learning: path within the extracted zip (uses {L})
    sae_release: Optional[str] = None         # HF repo id (sae_lens and dictionary_learning)
    sae_release_filename: Optional[str] = None  # for dictionary_learning: zip filename inside the HF repo
    sae_hook_template: Optional[str] = None   # for sae_lens (uses {L})
    sae_repo: Optional[str] = None            # for sparsify
    # Analysis defaults
    threshold: float = 0.2
    site_template: str = "layer{L}"
    default_layers: Optional[list] = None     # None -> range(num_layers)


PRESETS: dict[str, Preset] = {
    # Current baseline: Pythia-70m + saprmarks dictionary_learning SAEs.
    # Weights auto-download from saprmarks/pythia-70m-deduped-saes on first use
    # (archive unpacks under the HF cache).
    "pythia-70m": Preset(
        name="pythia-70m",
        model_id="EleutherAI/pythia-70m-deduped",
        num_layers=6, d_model=512,
        embed_path="gpt_neox.embed_in",
        layer_path="gpt_neox.layers",
        sae_loader="dictionary_learning",
        sae_arch="relu",
        sae_release="saprmarks/pythia-70m-deduped-saes",
        sae_release_filename="dictionaries_pythia-70m-deduped_10.zip",
        sae_path_template="dictionaries/pythia-70m-deduped/resid_out_layer{L}/10_32768/ae.pt",
        threshold=0.2,
        site_template="resid_out_layer{L}",
    ),

    # GPT-2 Small + jbloom OpenAI v5 (ReLU, all 12 layers)
    "gpt2-small": Preset(
        name="gpt2-small",
        model_id="openai-community/gpt2",
        num_layers=12, d_model=768,
        embed_path="transformer.wte",
        layer_path="transformer.h",
        sae_loader="sae_lens",
        sae_arch="relu",
        sae_release="gpt2-small-resid-post-v5-32k",
        sae_hook_template="blocks.{L}.hook_resid_post",
        threshold=0.2,
        site_template="resid_post_layer{L}",
    ),

    # Gemma-2 2B + Gemma Scope (JumpReLU, all 26 layers). Canonical release is
    # the 16k-width residual-stream set.
    "gemma-2-2b": Preset(
        name="gemma-2-2b",
        model_id="google/gemma-2-2b",
        num_layers=26, d_model=2304,
        embed_path="model.embed_tokens",
        layer_path="model.layers",
        sae_loader="sae_lens",
        sae_arch="jumprelu",
        sae_release="gemma-scope-2b-pt-res-canonical",
        sae_hook_template="layer_{L}/width_16k/canonical",
        threshold=0.0,  # JumpReLU has its own per-feature threshold baked in
        site_template="resid_layer{L}",
    ),

    # Llama-3.2-1B + chanind SAELens (JumpReLU typical; auto-detected at load
    # from sae.cfg.architecture). Covers layers 0-8.
    "llama-3.2-1b": Preset(
        name="llama-3.2-1b",
        model_id="unsloth/Llama-3.2-1B",
        num_layers=16, d_model=2048,
        embed_path="model.embed_tokens",
        layer_path="model.layers",
        sae_loader="sae_lens",
        sae_arch="jumprelu",
        sae_release="chanind/sae-llama-3.2-1b-res",
        sae_hook_template="blocks.{L}.hook_resid_post",
        threshold=0.0,
        site_template="resid_layer{L}",
        default_layers=list(range(9)),  # chanind covers L0-L8
    ),

    # Llama-3-8B (Meta gated model) + EleutherAI sparsify TopK SAEs.
    # Repo covers layers.0..layers.30 (31 of 32) plus embed_tokens.
    # Each SAE is ~4 GB fp32; do NOT load all at once -- submit layer subsets.
    "llama-3-8b": Preset(
        name="llama-3-8b",
        model_id="meta-llama/Meta-Llama-3-8B",
        num_layers=32, d_model=4096,
        embed_path="model.embed_tokens",
        layer_path="model.layers",
        sae_loader="sparsify",
        sae_arch="topk",
        sae_repo="EleutherAI/sae-llama-3-8b-32x",
        sae_hook_template="layers.{L}",
        threshold=0.0,  # TopK has its own selection; threshold on the post-TopK output
        site_template="resid_layer{L}",
        # A representative subset avoiding full 31-layer memory blowup.
        default_layers=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    ),

    # Qwen2-0.5B + chanind (SAELens, all 24 layers)
    "qwen2-0.5b": Preset(
        name="qwen2-0.5b",
        model_id="Qwen/Qwen2-0.5B",
        num_layers=24, d_model=896,
        embed_path="model.embed_tokens",
        layer_path="model.layers",
        sae_loader="sae_lens",
        sae_arch="jumprelu",
        sae_release="chanind/sae-qwen2-0.5b-res",
        sae_hook_template="blocks.{L}.hook_resid_post",
        threshold=0.0,
        site_template="resid_post_layer{L}",
        default_layers=list(range(12)),  # chanind covers L0-L11 only
    ),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        avail = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown preset {name!r}. Available: {avail}")
    return PRESETS[name]


def list_presets() -> list[str]:
    return sorted(PRESETS.keys())


def site_for(preset: Preset, layer_idx: int) -> str:
    return preset.site_template.format(L=layer_idx)
