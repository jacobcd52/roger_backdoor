from __future__ import annotations

from typing import List, Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RunConfig, get_torch_dtype


def load_model_and_tokenizer(cfg: RunConfig):
    dtype = get_torch_dtype(cfg)
    tok = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
    )
    return model, tok


def get_transformer_layers(model: Any) -> List[Any]:
    # Supports common architectures (Llama/Mistral/GPT-NeoX/GPT2 variants)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    # Fallback to trying common names
    for name in ["layers", "h", "blocks"]:
        if hasattr(model, name):
            seq = getattr(model, name)
            try:
                return list(seq)
            except Exception:
                pass
    raise RuntimeError("Could not locate transformer layers on model; unsupported architecture for residual hook")
