from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import RunConfig, load_config
from .model import load_model_and_tokenizer
from .utils import read_jsonl, append_jsonl


def _token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # logits: [1, T, V], input_ids: [1, T]
    logp = F.log_softmax(logits, dim=-1)
    tgt = input_ids[:, 1:]  # predict token t given t-1
    pred_logp = logp[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    return pred_logp.squeeze(0)  # [T-1]


def _get_model_device(model) -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def evaluate_run(run_dir: str, cfg: RunConfig) -> str:
    model_name = cfg.reference_model_name or cfg.model_name
    ref_cfg = RunConfig(**{**cfg.to_dict(), "model_name": model_name})
    model, tok = load_model_and_tokenizer(ref_cfg)
    model.eval()
    device = _get_model_device(model)

    gens_paths = sorted(glob.glob(os.path.join(run_dir, "generations", "gens_noise_*.jsonl")))
    out_path = os.path.join(run_dir, "eval_logprobs.jsonl")

    for gp in tqdm(gens_paths, desc="Evaluating"):
        gens = read_jsonl(gp)
        recs: List[Dict] = []
        for rec in gens:
            messages = rec["messages"]
            prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            assistant_text = rec["assistant_text"]

            prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(dtype=torch.long, device=device)
            assist_ids = tok(assistant_text, return_tensors="pt", add_special_tokens=False).input_ids.to(dtype=torch.long, device=device)
            full_ids = torch.cat([prompt_ids, assist_ids], dim=1)

            with torch.no_grad():
                out = model(full_ids)
                logits = out.logits  # [1, T, V]
            token_logp = _token_logprobs(logits, full_ids)  # [T-1]
            prompt_len = prompt_ids.size(1)
            # Assistant token positions start at index prompt_len
            assist_logp = token_logp[prompt_len - 1 : prompt_len - 1 + assist_ids.size(1)]
            total_logprob = float(assist_logp.sum().item())

            new_rec = {
                "noise_id": rec.get("noise_id"),
                "noise_scale": rec.get("noise_scale"),
                "prompt_index": rec.get("prompt_index"),
                "total_logprob": total_logprob,
                "per_token_logprobs": [float(x) for x in assist_logp.tolist()],
                "assistant_text": assistant_text,
            }
            recs.append(new_rec)
        append_jsonl(out_path, recs)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    evaluate_run(args.run_dir, cfg)


if __name__ == "__main__":
    main()
