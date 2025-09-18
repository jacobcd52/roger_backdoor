from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .config import RunConfig, load_config, ensure_run_dir, save_config, get_torch_dtype
from .data import load_prompt_messages, apply_chat_templates
from .model import load_model_and_tokenizer
from .hooks import ResidualVectorAdder
from .utils import (
    seed_all,
    ensure_dir,
    chunked,
    append_jsonl,
    save_numpy,
)


def _batch_decode_assistant_only(tokenizer, sequences: torch.Tensor, original_padded_len: int) -> List[str]:
    decoded: List[str] = []
    for i in range(sequences.size(0)):
        gen_ids = sequences[i, original_padded_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        decoded.append(text)
    return decoded


def _maybe_load_cov_sqrt(base_output_dir: str, model_name: str, layer_index: int) -> np.ndarray | None:
    # Look for covariance file directly in cov/ directory
    cov_dir = os.path.join(base_output_dir, "cov")
    if os.path.isdir(cov_dir):
        model_name_safe = model_name.replace("/", "_").replace("-", "_")
        layer_str = f"layer{layer_index}"
        cov_sqrt_name = f"cov_sqrt_{model_name_safe}_{layer_str}.npy"
        path = os.path.join(cov_dir, cov_sqrt_name)
        if os.path.exists(path):
            try:
                return np.load(path)
            except Exception:
                pass
    return None


def run(cfg: RunConfig, config_path: str) -> str:
    seed_all(cfg.seed)

    run_dir = ensure_run_dir(cfg.output_dir)
    gens_dir = ensure_dir(os.path.join(run_dir, "generations"))
    noise_dir = ensure_dir(os.path.join(run_dir, "noises"))

    saved_cfg_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    with open(saved_cfg_path, "w", encoding="utf-8") as f:
        f.write(raw)

    model, tokenizer = load_model_and_tokenizer(cfg)

    # Use left padding so generated tokens appear strictly to the right
    tokenizer.padding_side = "left"

    messages_list = load_prompt_messages(cfg.dataset_name, cfg.num_prompts)
    prompts = apply_chat_templates(tokenizer, messages_list, add_generation_prompt=True)

    # Filter prompts by max_prompt_len (tokenized length without truncation)
    max_prompt_len = getattr(cfg, "max_prompt_len", 256) or 256
    tokenized_all = tokenizer(
        prompts,
        add_special_tokens=False,
        return_attention_mask=False,
        padding=False,
    )
    keep_indices: List[int] = [
        i for i, ids in enumerate(tokenized_all["input_ids"]) if len(ids) <= max_prompt_len
    ]
    if len(keep_indices) != len(prompts):
        prompts = [prompts[i] for i in keep_indices]
        messages_list = [messages_list[i] for i in keep_indices]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-tokenize all prompts with left padding
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=bool(cfg.max_input_length),
        max_length=cfg.max_input_length or None,
    )
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    # With left padding, all rows share the same padded length
    original_padded_len = int(input_ids.size(1))

    hidden_size = int(getattr(model.config, "hidden_size", input_ids.size(-1)))

    # Optional covariance shaping
    cov_sqrt = _maybe_load_cov_sqrt(cfg.output_dir, cfg.model_name, cfg.layer_index)
    if cov_sqrt is not None and cov_sqrt.shape == (hidden_size, hidden_size):
        cov_sqrt_mat = cov_sqrt.astype(np.float32)
    else:
        cov_sqrt_mat = None

    scales: List[float] = (
        list(cfg.noise_norms) if getattr(cfg, "noise_norms", None) else [float(cfg.noise_norm)]
    )

    for noise_idx in range(cfg.num_noise_vectors):
        rng = np.random.default_rng(cfg.seed + noise_idx)
        base_vec = rng.normal(size=(hidden_size,)).astype(np.float32)
        base_vec /= (np.linalg.norm(base_vec) + 1e-12)  # normalize to unit sphere
        if cov_sqrt_mat is not None:
            # Apply covariance sqrt transformation
            base_vec = cov_sqrt_mat @ base_vec

        vec_path = os.path.join(noise_dir, f"noise_{noise_idx:04d}.npy")
        save_numpy(vec_path, base_vec)

        out_path = os.path.join(gens_dir, f"gens_noise_{noise_idx:04d}.jsonl")

        for scale in scales:
            scaled_vec = (base_vec * float(scale)).astype(np.float32)
            hook = ResidualVectorAdder(model, cfg.layer_index, torch.from_numpy(scaled_vec))
            hook.register()

            for idx_batch in tqdm(
                list(chunked(list(range(len(prompts))), cfg.batch_size)),
                desc=f"Noise {noise_idx} scale {scale}",
            ):
                batch_input_ids = input_ids[idx_batch].to(device)
                batch_attention_mask = attention_mask[idx_batch].to(device)

                with torch.no_grad():
                    sequences = model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        max_new_tokens=cfg.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                # Assistant-only tokens are strictly beyond the original padded length
                batch_generations = _batch_decode_assistant_only(
                    tokenizer, sequences, original_padded_len
                )

                records: List[Dict] = []
                for bi, global_i in enumerate(idx_batch):
                    rec = {
                        "assistant_text": batch_generations[bi],
                        "noise_id": noise_idx,
                        "noise_scale": float(scale),
                        "prompt_index": int(global_i),
                        "messages": messages_list[global_i],
                        "prompt_text": prompts[global_i],
                    }
                    records.append(rec)

                append_jsonl(out_path, records)

            hook.remove()

    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = run(cfg, args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
