from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .utils import read_jsonl


def _colorize_tokens(tokens: List[str], logprobs: List[float]) -> Text:
    # Map logprob to a red intensity where more negative is darker
    if not tokens:
        return Text("")
    lp_min = min(logprobs)
    lp_max = max(logprobs)
    span = max(1e-6, lp_max - lp_min)
    text = Text()
    for tok, lp in zip(tokens, logprobs):
        # Normalize so lowest is 1, highest is 0
        t = (lp_max - lp) / span
        r = int(255 * min(1.0, 0.3 + 0.7 * t))
        g = int(255 * (1.0 - 0.7 * t))
        b = int(255 * (1.0 - 0.7 * t))
        style = f"bold rgb({r},{g},{b})"
        text.append(tok, style=style)
    return text


def show_worst(run_dir: str, top_k: int = 10) -> None:
    eval_path = os.path.join(run_dir, "eval_logprobs.jsonl")
    recs = read_jsonl(eval_path)
    recs_sorted = sorted(recs, key=lambda r: r["total_logprob"])[:top_k]

    table = Table(title="Worst total logprobs")
    table.add_column("noise_id")
    table.add_column("scale")
    table.add_column("prompt_index")
    table.add_column("total_logprob")
    table.add_column("assistant_text")

    for r in recs_sorted:
        table.add_row(
            str(r.get("noise_id")),
            f"{r.get('noise_scale', 0.0)}",
            str(r.get("prompt_index")),
            f"{r['total_logprob']:.3f}",
            r["assistant_text"][:200],
        )

    Console().print(table)


def show_example(run_dir: str, noise_id: int, prompt_index: int) -> None:
    eval_path = os.path.join(run_dir, "eval_logprobs.jsonl")
    gen_paths = glob.glob(os.path.join(run_dir, "generations", f"gens_noise_{noise_id:04d}.jsonl"))
    if not gen_paths:
        raise FileNotFoundError("Generations file not found for given noise_id")

    eval_recs = [r for r in read_jsonl(eval_path) if r["noise_id"] == noise_id and r["prompt_index"] == prompt_index]
    if not eval_recs:
        raise ValueError("Evaluation record not found for given noise_id and prompt_index")
    eval_rec = eval_recs[0]

    gen_recs = [r for r in read_jsonl(gen_paths[0]) if r["prompt_index"] == prompt_index]
    if not gen_recs:
        raise ValueError("Generation record not found for given prompt_index")
    gen_rec = gen_recs[0]

    # Token-level display by naively splitting; better view is at character level
    tokens = list(gen_rec["assistant_text"])  # character-wise fallback for display
    token_text = _colorize_tokens(tokens, eval_rec["per_token_logprobs"]) if eval_rec["per_token_logprobs"] else Text(gen_rec["assistant_text"]) 

    Console().print(
        f"noise_id={noise_id} scale={eval_rec.get('noise_scale')} prompt_index={prompt_index} total_logprob={eval_rec['total_logprob']:.3f}"
    )
    Console().print(token_text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--noise_id", type=int)
    ap.add_argument("--prompt_index", type=int)
    args = ap.parse_args()

    if args.noise_id is not None and args.prompt_index is not None:
        show_example(args.run_dir, args.noise_id, args.prompt_index)
    else:
        show_worst(args.run_dir, args.top_k)


if __name__ == "__main__":
    main()
