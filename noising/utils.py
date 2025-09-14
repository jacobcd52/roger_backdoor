from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple, Dict, Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch  # local import to avoid hard dep at import time
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def chunked(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    if size <= 0:
        size = 1
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def save_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_numpy(path: str, array: np.ndarray) -> None:
    ensure_dir(str(Path(path).parent))
    np.save(path, array)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec) + 1e-12
    return vec / denom


def make_noise_vector(hidden_size: int, norm: float, rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=(hidden_size,)).astype(np.float32)
    vec = l2_normalize(vec) * float(norm)
    return vec


def detect_device() -> str:
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            return "cuda"
        if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
