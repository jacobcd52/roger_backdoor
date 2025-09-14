from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List
import os
import time
import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class RunConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    reference_model_name: Optional[str] = None
    dtype: str = "bf16"
    dataset_name: str = "lmsys/lmsys-chat-1m"
    num_prompts: int = 1000
    batch_size: int = 4
    max_new_tokens: int = 128
    layer_index: int = 10
    noise_norm: float = 1.0
    noise_norms: Optional[List[float]] = None
    num_noise_vectors: int = 8
    seed: int = 0
    output_dir: str = "noising/outputs"
    trust_remote_code: bool = True
    device_map: str = "auto"
    max_input_length: Optional[int] = None
    max_prompt_len: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


def _map_dtype_name_to_torch(dtype_name: str):
    name = (dtype_name or "").lower()
    if torch is None:
        return None
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = RunConfig(**data)
    # Robust defaults for potentially null entries
    if getattr(cfg, "num_noise_vectors", None) is None:
        cfg.num_noise_vectors = 8
    if getattr(cfg, "max_prompt_len", None) in (None, 0):
        cfg.max_prompt_len = 256
    # Normalize noise_norms: if empty list, treat as None
    if getattr(cfg, "noise_norms", None) is not None and len(cfg.noise_norms) == 0:  # type: ignore[arg-type]
        cfg.noise_norms = None
    return cfg


def save_config(cfg: RunConfig, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"config_{ts}.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    return out_path


def ensure_run_dir(base_dir: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, ts)
    os.makedirs(path, exist_ok=True)
    return path


def get_torch_dtype(cfg: RunConfig):
    return _map_dtype_name_to_torch(cfg.dtype)
