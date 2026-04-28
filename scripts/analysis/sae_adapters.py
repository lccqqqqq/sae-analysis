"""
Uniform SAE loading + encoder API across source libraries and activation types.

One entry point, `load_sae(preset, layer_idx, device)`, returns an `SAEBundle`
that exposes `.encode(x)` regardless of whether the SAE is a vanilla ReLU
(dictionary_learning), a JumpReLU (Gemma Scope / chanind via sae_lens), or a
TopK (EleutherAI via sparsify).

The analysis pipeline calls `bundle.encode(x)` and never branches on arch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from presets import Preset


@dataclass
class SAEBundle:
    """A loaded SAE, normalized to a common format.

    Shape conventions:
        enc_w: [d_model, n_latent]    — so  preact = x @ enc_w + enc_b
        enc_b: [n_latent]
        dec_w: [d_model, n_latent]    — so  recon = f @ dec_w.T  (or dec_w @ f on a single vector)
        dec_b: [d_model] or None
        threshold_vec: [n_latent] or None (for JumpReLU)
        topk: int or None              (for TopK)
    """

    enc_w: torch.Tensor
    enc_b: torch.Tensor
    dec_w: torch.Tensor
    dec_b: Optional[torch.Tensor]
    arch: str
    d_model: int
    n_latent: int
    threshold_vec: Optional[torch.Tensor] = None
    topk: Optional[int] = None
    source: str = ""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute post-activation feature values. Shape-preserving on the last dim:
        input [..., d_model] -> output [..., n_latent]."""
        preact = torch.matmul(x, self.enc_w) + self.enc_b
        if self.arch == "relu":
            return F.relu(preact)
        if self.arch == "jumprelu":
            # Mask is detached (boolean); gradient flows only through surviving
            # preactivations, matching the standard JumpReLU STE convention.
            mask = (preact > self.threshold_vec).to(preact.dtype)
            return preact * mask
        if self.arch == "topk":
            k = self.topk
            assert k is not None
            # Zero out all but the top-k preactivations along the last dim.
            topk_vals, topk_idx = torch.topk(preact, k=k, dim=-1)
            out = torch.zeros_like(preact)
            out.scatter_(-1, topk_idx, F.relu(topk_vals))
            return out
        raise ValueError(f"Unknown SAE architecture: {self.arch!r}")


# --- Loader 1: dictionary_learning (.pt state dict from zipped HF release) --

def _load_dictionary_learning(
    preset: Preset, layer_idx: int, device: str
) -> SAEBundle:
    if not (preset.sae_release and preset.sae_release_filename and preset.sae_path_template):
        raise ValueError(
            f"Preset {preset.name!r} uses dictionary_learning loader but "
            "is missing sae_release / sae_release_filename / sae_path_template."
        )

    from huggingface_hub import hf_hub_download
    import zipfile

    zip_path = Path(hf_hub_download(
        repo_id=preset.sae_release, filename=preset.sae_release_filename,
    ))
    extract_root = zip_path.parent / "extracted"
    path = extract_root / preset.sae_path_template.format(L=layer_idx)
    if not path.exists():
        print(f"[INFO] Extracting {zip_path.name} -> {extract_root}", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)

    print(f"[INFO] Loading dictionary_learning SAE from {path}", flush=True)
    sd = torch.load(path, map_location="cpu")

    d_model = preset.d_model

    # Same logic as the legacy get_sae_weights:
    dec_w = None
    for k, v in sd.items():
        if "decoder.weight" in k or "dec.weight" in k:
            dec_w = v
            break
    if dec_w is None:
        for v in sd.values():
            if v.ndim == 2 and (v.shape[0] == d_model or v.shape[1] == d_model):
                dec_w = v
                break
    if dec_w is None:
        raise ValueError("Could not find decoder weights in state dict")
    if dec_w.shape[0] != d_model:
        dec_w = dec_w.T

    enc_w = sd.get("encoder.weight")
    if enc_w is None:
        enc_w = dec_w.T
    else:
        if enc_w.shape[1] == d_model:
            enc_w = enc_w.T

    enc_b = sd.get("encoder.bias")
    if enc_b is None:
        enc_b = torch.zeros(dec_w.shape[1])

    dec_b = sd.get("decoder.bias") or sd.get("bias")  # often absent

    n_latent = dec_w.shape[1]
    return SAEBundle(
        enc_w=enc_w.to(device),
        enc_b=enc_b.to(device),
        dec_w=dec_w.to(device),
        dec_b=dec_b.to(device) if isinstance(dec_b, torch.Tensor) else None,
        arch="relu",
        d_model=d_model,
        n_latent=n_latent,
        source=str(path),
    )


# --- Loader 2: sae_lens ------------------------------------------------------

def _load_sae_lens(preset: Preset, layer_idx: int, device: str) -> SAEBundle:
    try:
        from sae_lens import SAE  # type: ignore
    except ImportError as e:
        raise ImportError(
            "sae_lens is required for this preset. Install with `uv pip install sae-lens`."
        ) from e

    if preset.sae_release is None or preset.sae_hook_template is None:
        raise ValueError(
            f"Preset {preset.name!r} missing sae_release/sae_hook_template."
        )
    hook = preset.sae_hook_template.format(L=layer_idx)
    print(f"[INFO] Loading sae_lens SAE: release={preset.sae_release} hook={hook}",
          flush=True)
    try:
        sae, _cfg_dict, _sparsity = SAE.from_pretrained(
            release=preset.sae_release, sae_id=hook, device=device
        )
    except (KeyError, ValueError) as e:
        # Release not in sae_lens directory — fall back to direct HF download.
        # Works for any repo with the standard layout: <hook>/cfg.json +
        # <hook>/sae_weights.safetensors.
        from huggingface_hub import snapshot_download
        print(f"[INFO] Release not registered ({e}); downloading from HF "
              f"{preset.sae_release} subfolder {hook}...", flush=True)
        local_dir = snapshot_download(
            repo_id=preset.sae_release, allow_patterns=[f"{hook}/*"],
        )
        sae = SAE.load_from_pretrained(f"{local_dir}/{hook}", device=device)

    # Architecture detection from the loaded config (more authoritative than
    # the preset default, which is only a hint). In sae_lens >= 6.x,
    # cfg.architecture is a classmethod returning a string; call if callable.
    arch_from_cfg = getattr(sae.cfg, "architecture", None)
    if callable(arch_from_cfg):
        try:
            arch_from_cfg = arch_from_cfg()
        except Exception:
            arch_from_cfg = None
    # Fallback: introspect the cfg class name.
    if not isinstance(arch_from_cfg, str):
        cfg_cls = type(sae.cfg).__name__.lower()
        if "jumprelu" in cfg_cls:
            arch_from_cfg = "jumprelu"
        elif "topk" in cfg_cls:
            arch_from_cfg = "topk"
        elif "standard" in cfg_cls or "gated" in cfg_cls:
            arch_from_cfg = "standard"
    if arch_from_cfg == "jumprelu":
        arch = "jumprelu"
    elif arch_from_cfg in ("standard", "gated"):
        arch = "relu"
    elif arch_from_cfg == "topk":
        arch = "topk"
    else:
        arch = preset.sae_arch  # last-resort fallback
    print(f"[INFO]   detected arch={arch} (cfg said {arch_from_cfg!r})", flush=True)

    # Extract tensors. SAELens stores W_enc as [d_in, d_sae] and W_dec as
    # [d_sae, d_in]. Normalize to our convention:
    #   enc_w: [d_model, n_latent]
    #   dec_w: [d_model, n_latent]
    enc_w = sae.W_enc.detach()
    dec_w_raw = sae.W_dec.detach()
    if dec_w_raw.shape[0] != enc_w.shape[0]:
        dec_w = dec_w_raw.T.contiguous()
    else:
        dec_w = dec_w_raw
    enc_b = sae.b_enc.detach()
    dec_b = sae.b_dec.detach() if hasattr(sae, "b_dec") else None

    threshold_vec = None
    if arch == "jumprelu":
        # SAELens exposes the threshold as `threshold` (raw) or `log_threshold`.
        if hasattr(sae, "threshold"):
            threshold_vec = sae.threshold.detach()
        elif hasattr(sae, "log_threshold"):
            threshold_vec = torch.exp(sae.log_threshold.detach())
        else:
            raise ValueError("JumpReLU SAE has no `threshold` attribute.")

    topk = None
    if arch == "topk":
        topk = getattr(sae.cfg, "k", None) or getattr(sae.cfg, "topk", None)
        if topk is None:
            raise ValueError("TopK SAE has no `k` in cfg.")

    d_model = enc_w.shape[0]
    n_latent = enc_w.shape[1]

    bundle = SAEBundle(
        enc_w=enc_w.to(device),
        enc_b=enc_b.to(device),
        dec_w=dec_w.to(device),
        dec_b=dec_b.to(device) if dec_b is not None else None,
        arch=arch,
        d_model=d_model,
        n_latent=n_latent,
        threshold_vec=threshold_vec.to(device) if threshold_vec is not None else None,
        topk=topk,
        source=f"sae_lens:{preset.sae_release}/{hook}",
    )
    return bundle


# --- Loader 3: sparsify (EleutherAI) -----------------------------------------

def _load_sparsify(preset: Preset, layer_idx: int, device: str) -> SAEBundle:
    """Load EleutherAI sparsify-format SAEs by reading safetensors directly.

    The sparsify python package fails to build in some envs (onnx wheel issue),
    but the on-disk format is simple: per-hookpoint subdir with cfg.json
    + sae.safetensors. We download and parse them ourselves.

    Format (as of EleutherAI/sae-llama-3-8b-32x):
        layers.<L>/cfg.json           -- {"d_in", "expansion_factor", "k", ...}
        layers.<L>/sae.safetensors    -- encoder.weight [n_latent, d_in],
                                         encoder.bias   [n_latent],
                                         W_dec          [n_latent, d_in],
                                         b_dec          [d_in]
    """
    import json
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    if preset.sae_repo is None or preset.sae_hook_template is None:
        raise ValueError(
            f"Preset {preset.name!r} missing sae_repo/sae_hook_template."
        )
    hook = preset.sae_hook_template.format(L=layer_idx)
    print(f"[INFO] Loading sparsify SAE (direct safetensors): "
          f"repo={preset.sae_repo} hook={hook}", flush=True)

    cfg_path = hf_hub_download(preset.sae_repo, f"{hook}/cfg.json")
    st_path = hf_hub_download(preset.sae_repo, f"{hook}/sae.safetensors")
    cfg = json.load(open(cfg_path))

    tensors = {}
    with safe_open(st_path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    # Cast to float32 on the target device. Adapter convention:
    # enc_w shape [d_model, n_latent] (so preact = x @ enc_w + enc_b),
    # dec_w shape [d_model, n_latent].
    enc_w = tensors["encoder.weight"].to(device=device, dtype=torch.float32).T.contiguous()
    enc_b = tensors["encoder.bias"].to(device=device, dtype=torch.float32)
    dec_w = tensors["W_dec"].to(device=device, dtype=torch.float32).T.contiguous()
    dec_b = (tensors["b_dec"].to(device=device, dtype=torch.float32)
             if "b_dec" in tensors else None)

    topk = int(cfg.get("k")) if cfg.get("k") is not None else None
    bundle = SAEBundle(
        enc_w=enc_w,
        enc_b=enc_b,
        dec_w=dec_w,
        dec_b=dec_b,
        arch="topk",
        d_model=enc_w.shape[0],
        n_latent=enc_w.shape[1],
        topk=topk,
        source=f"sparsify-direct:{preset.sae_repo}/{hook}",
    )
    return bundle


# --- Public entry point ------------------------------------------------------

def load_sae(preset: Preset, layer_idx: int, device: str) -> SAEBundle:
    """Load an SAE for (preset, layer) and return a uniform SAEBundle."""
    loader = preset.sae_loader
    if loader == "dictionary_learning":
        return _load_dictionary_learning(preset, layer_idx, device)
    if loader == "sae_lens":
        return _load_sae_lens(preset, layer_idx, device)
    if loader == "sparsify":
        return _load_sparsify(preset, layer_idx, device)
    raise ValueError(f"Unknown sae_loader {loader!r}")
