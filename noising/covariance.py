from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from .config import RunConfig, load_config, ensure_run_dir
from .data import load_prompt_messages, apply_chat_templates
from .model import load_model_and_tokenizer, get_transformer_layers
from .utils import ensure_dir


class ResidualCollector:
    def __init__(self, model, layer_index: int):
        self.layer_index = int(layer_index)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.model = model
        # Running statistics for Welford's algorithm
        self.n = 0
        self.mean = None
        self.M2 = None  # Sum of squared differences from mean

    def _post_hook(self, module: torch.nn.Module, args, kwargs, output):  # type: ignore[override]
        # output: hidden_states [batch, seq, d]
        try:
            hs = output
            if isinstance(hs, (tuple, list)):
                hs = hs[0]
            # take last token per sequence
            last = hs[:, -1, :].detach().cpu().float()  # [batch, d]
            # Update running statistics
            for i in range(last.size(0)):
                x = last[i].numpy()  # [d]
                self._update_running_stats(x)
        except Exception:
            pass
        return output

    def _update_running_stats(self, x: np.ndarray):
        """Update running mean and covariance using Welford's algorithm"""
        self.n += 1
        if self.n == 1:
            self.mean = x.copy()
            self.M2 = np.zeros((len(x), len(x)))  # Covariance matrix
        else:
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            # Update covariance: M2 += delta * delta2^T (outer product)
            self.M2 += np.outer(delta, delta2)

    def get_covariance(self) -> np.ndarray:
        """Get the final covariance matrix"""
        if self.n < 2:
            raise RuntimeError("Need at least 2 samples to compute covariance")
        # Unbiased estimator: divide by (n-1)
        return self.M2 / (self.n - 1)

    def register(self):
        layers = get_transformer_layers(self.model)
        if not (0 <= self.layer_index < len(layers)):
            raise IndexError(f"layer_index {self.layer_index} out of range")
        target = layers[self.layer_index]
        h = target.register_forward_hook(self._post_hook, with_kwargs=True)
        self.handles.append(h)

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()


# Note: compute_covariance function removed as we now use running statistics


def matrix_sqrt_via_svd(C: np.ndarray) -> np.ndarray:
    # Convert to GPU tensor for efficient SVD computation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C_tensor = torch.from_numpy(C).float().to(device)
    
    # Compute SVD on GPU
    U, S, Vt = torch.linalg.svd(C_tensor, full_matrices=False)
    S_sqrt = torch.sqrt(torch.clamp(S, min=0.0))
    result = (U * S_sqrt.unsqueeze(0)) @ Vt
    
    # Convert back to numpy
    return result.cpu().numpy()


def run(cfg: RunConfig, config_path: str, out_dir: str | None = None) -> str:
    model, tokenizer = load_model_and_tokenizer(cfg)
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

    # Tokenize a subset to form 10 batches
    batch_size = cfg.batch_size
    num_batches = 20
    max_items = batch_size * num_batches
    prompts = prompts[:max_items]

    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=bool(cfg.max_input_length),
        max_length=cfg.max_input_length or None,
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    collector = ResidualCollector(model, cfg.layer_index)
    collector.register()

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Collecting"):
            s = i * batch_size
            e = s + batch_size
            ids = input_ids[s:e]
            am = attention_mask[s:e]
            
            # Generate responses and truncate after max_new_tokens
            try:
                sequences = model.generate(
                    input_ids=ids,
                    attention_mask=am,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,  # Deterministic generation
                )
                
                # Create attention mask for generated sequences
                gen_attention_mask = torch.ones_like(sequences)
                
                # Forward pass the truncated sequences to collect residuals
                _ = model(input_ids=sequences, attention_mask=gen_attention_mask)
                
            except Exception as e:
                print(f"Warning: Batch {i} failed: {e}")
                continue

    collector.remove()

    if collector.n < 2:
        raise RuntimeError("No residuals collected; check layer_index and model architecture")

    # Get covariance from running statistics
    C = collector.get_covariance()
    C_sqrt = matrix_sqrt_via_svd(C)

    # Save with informative name (no timestamp)
    base_dir = out_dir or cfg.output_dir
    cov_dir = os.path.join(base_dir, "cov")
    ensure_dir(cov_dir)
    model_name_safe = cfg.model_name.replace("/", "_").replace("-", "_")
    layer_str = f"layer{cfg.layer_index}"
    cov_name = f"cov_{model_name_safe}_{layer_str}.npy"
    cov_sqrt_name = f"cov_sqrt_{model_name_safe}_{layer_str}.npy"
    np.save(os.path.join(cov_dir, cov_name), C)
    np.save(os.path.join(cov_dir, cov_sqrt_name), C_sqrt)
    return cov_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out_dir", type=str)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = run(cfg, args.config, args.out_dir)
    print(out)


if __name__ == "__main__":
    main()
