"""
Model wiring helpers that dispatch by dotted-path strings from a Preset.

The analysis pipeline needs three model-specific hooks:
  1. Read the embedding module (to embed inputs ahead of the forward pass).
  2. Overwrite the embedding module with a DummyEmbed (to route a leaf tensor
     with requires_grad=True through the forward pass).
  3. Pick the residual-stream module at layer L to attach a forward hook.

Rather than branching on model family, the Preset carries the dotted attribute
paths and these helpers resolve them.
"""

from __future__ import annotations

from functools import reduce
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from presets import Preset


def _resolve(root: Any, dotted: str) -> Any:
    return reduce(getattr, dotted.split("."), root)


def _set_attr_by_path(root: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    parent = reduce(getattr, parts[:-1], root) if len(parts) > 1 else root
    setattr(parent, parts[-1], value)


def load_model(preset: Preset, device: str) -> tuple[Any, Any]:
    """Load and move the HF model + tokenizer specified by the preset."""
    print(f"[INFO] Loading model {preset.model_id} on {device}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(preset.model_id).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(preset.model_id)
    return model, tokenizer


def get_embed(model: Any, preset: Preset) -> Any:
    return _resolve(model, preset.embed_path)


def set_embed(model: Any, preset: Preset, new_module: Any) -> None:
    _set_attr_by_path(model, preset.embed_path, new_module)


def get_layer(model: Any, preset: Preset, layer_idx: int) -> Any:
    layers = _resolve(model, preset.layer_path)
    return layers[layer_idx]


class DummyEmbed(torch.nn.Module):
    """Passthrough module that returns a pre-computed embeddings tensor.

    Used to swap out the model's embedding layer during the gradient pipeline
    so that a leaf tensor with requires_grad=True flows through the forward
    pass. The input_ids argument is ignored; the tensor is pre-built outside.
    """

    def __init__(self, embeds: torch.Tensor):
        super().__init__()
        self.embeds = embeds

    def forward(self, input_ids):  # noqa: ARG002  (unused, by design)
        return self.embeds
