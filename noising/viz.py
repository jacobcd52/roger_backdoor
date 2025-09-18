from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Tuple

from transformers import AutoTokenizer

from .config import RunConfig, load_config
from .utils import read_jsonl


def _load_tokenizer(cfg: RunConfig):
    name = cfg.reference_model_name or cfg.model_name
    tok = AutoTokenizer.from_pretrained(
        name, use_fast=True, trust_remote_code=cfg.trust_remote_code
    )
    return tok


essential_fields = [
    "noise_id",
    "noise_scale",
    "prompt_index",
    "total_logprob",
    "assistant_text",
    "per_token_logprobs",
]


def _build_prompt_index(run_dir: str, noise_id: int, cache: Dict[int, Dict[int, Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
    if noise_id in cache:
        return cache[noise_id]
    path = os.path.join(run_dir, "generations", f"gens_noise_{noise_id:04d}.jsonl")
    index: Dict[int, Dict[str, Any]] = {}
    for rec in read_jsonl(path):
        idx = int(rec["prompt_index"]) if "prompt_index" in rec else int(rec.get("index", 0))
        index[idx] = rec
    cache[noise_id] = index
    return index


def _prepare_items_for_scale(
    recs: List[Dict[str, Any]],
    run_dir: str,
    tok,
    top_n: int,
) -> List[Dict[str, Any]]:
    # Sort by total_logprob ascending and take top_n
    recs_sorted = sorted(recs, key=lambda r: r["total_logprob"])[:top_n]

    gens_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}
    out: List[Dict[str, Any]] = []

    for r in recs_sorted:
        noise_id = int(r.get("noise_id", -1))
        prompt_index = int(r.get("prompt_index", -1))
        # Fetch prompt_text via generation index
        gen_index = _build_prompt_index(run_dir, noise_id, gens_cache)
        prompt_text = gen_index.get(prompt_index, {}).get("prompt_text", "")

        assistant_text = r.get("assistant_text", "")
        per_token_lps: List[float] = r.get("per_token_logprobs", []) or []

        tokens: List[str] = []
        lps: List[float] = []

        if per_token_lps:
            # Tokenize assistant_text to tokens with offsets for highlighting
            tokenized = tok(
                assistant_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = tokenized.get("offset_mapping")

            pairs: List[Tuple[int, int]] = []
            if isinstance(offsets, list) and len(offsets) > 0:
                first = offsets[0]
                # Case A: single input -> offsets is List[Tuple[int,int]]
                if isinstance(first, (list, tuple)) and len(first) == 2 and all(isinstance(x, int) for x in first):
                    pairs = [(int(s), int(e)) for (s, e) in offsets]  # type: ignore[arg-type]
                # Case B: nested batch -> offsets[0] is List[Tuple[int,int]]
                elif isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
                    pairs = [(int(s), int(e)) for (s, e) in first]  # type: ignore[misc]
            if pairs:
                tokens = [assistant_text[s:e] for (s, e) in pairs]
            else:
                # Fallback: character-wise split
                tokens = [assistant_text[i : i + 1] for i in range(len(assistant_text))]

            # Align lengths (truncate to min)
            L = min(len(tokens), len(per_token_lps))
            tokens = tokens[:L]
            lps = [float(x) for x in per_token_lps[:L]]

        # If no per-token logprobs or empty tokenization, fallback to whole text as one token
        if not tokens:
            tokens = [assistant_text]
            lps = [0.0]

        out.append(
            {
                "noise_id": noise_id,
                "noise_scale": r.get("noise_scale"),
                "prompt_index": prompt_index,
                "total_logprob": r.get("total_logprob"),
                "prompt_text": prompt_text,
                "assistant_tokens": [
                    {"text": t, "lp": float(lp)} for t, lp in zip(tokens, lps)
                ],
            }
        )

    return out


def _collect_by_scale(eval_path: str) -> Dict[str, List[Dict[str, Any]]]:
    recs = read_jsonl(eval_path)
    by_scale: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in recs:
        # Normalize scale key to string with limited precision to avoid float issues
        scale_val = r.get("noise_scale", 0.0)
        key = f"{float(scale_val):.6g}"
        # Keep only essential fields plus assistant_text and per_token_logprobs
        filtered = {k: r.get(k) for k in essential_fields}
        filtered["noise_scale"] = float(scale_val)
        by_scale[key].append(filtered)
    return dict(sorted(by_scale.items(), key=lambda kv: float(kv[0])))


def _render_html(data: Dict[str, Any]) -> str:
    # Minimal standalone HTML with embedded data and JS for dropdown + pagination + coloring
    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Noising Viz</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 16px; }}
.controls {{ margin-bottom: 12px; }}
.card {{ border: 1px solid #ddd; border-radius: 6px; padding: 12px; margin: 10px 0; }}
.meta {{ color: #555; font-size: 12px; margin-bottom: 6px; }}
.prompt {{ white-space: pre-wrap; color: #333; margin-bottom: 8px; }}
.assistant {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
.badge {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; margin-right: 6px; font-size: 12px; }}
.token {{ padding: 2px 1px; border-radius: 3px; }}
.pager button {{ margin-right: 6px; }}
</style>
</head>
<body>
<div class=\"controls\">
  <label for=\"scale\">Noise scale:</label>
  <select id=\"scale\"></select>
  <span id=\"count\"></span>
</div>
<div id=\"list\"></div>
<div class=\"pager\">
  <button id=\"prev\">Prev</button>
  <button id=\"next\">Next</button>
  <span id=\"pageinfo\"></span>
</div>
<script>
const DATA = {json.dumps(data)};
const PAGE_SIZE = 10;
let currentScaleKey = null;
let currentPage = 0;

function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text === null || text === undefined ? '' : String(text);
  return div.innerHTML;
}}

function colorFor(lp, lpMin, lpMax) {{
  // Map logprob to red tint: more negative -> darker red
  const span = Math.max(1e-6, lpMax - lpMin);
  const t = (lpMax - lp) / span; // 0..1
  const r = Math.floor(255 * Math.min(1.0, 0.3 + 0.7 * t));
  const g = Math.floor(255 * (1.0 - 0.7 * t));
  const b = Math.floor(255 * (1.0 - 0.7 * t));
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function renderList() {{
  const container = document.getElementById('list');
  container.innerHTML = '';
  if (!currentScaleKey) return;
  const items = DATA.itemsByScale[currentScaleKey] || [];
  const start = currentPage * PAGE_SIZE;
  const pageItems = items.slice(start, start + PAGE_SIZE);

  for (const it of pageItems) {{
    const card = document.createElement('div');
    card.className = 'card';

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = `<span class=\"badge\">noise_id: ${{escapeHtml(it.noise_id)}}</span>` +
                     `<span class=\"badge\">scale: ${{escapeHtml(it.noise_scale)}}</span>` +
                     `<span class=\"badge\">prompt_index: ${{escapeHtml(it.prompt_index)}}</span>` +
                     `<span class=\"badge\">total_logprob: ${{escapeHtml(Number(it.total_logprob).toFixed(3))}}</span>`;
    card.appendChild(meta);

    if (it.prompt_text) {{
      const prompt = document.createElement('div');
      prompt.className = 'prompt';
      prompt.innerText = it.prompt_text; // keep as text
      card.appendChild(prompt);
    }}

    const assistant = document.createElement('div');
    assistant.className = 'assistant';

    const lps = it.assistant_tokens.map(t => t.lp);
    const lpMin = lps.length ? Math.min(...lps) : 0.0;
    const lpMax = lps.length ? Math.max(...lps) : 0.0;

    for (const t of it.assistant_tokens) {{
      const span = document.createElement('span');
      span.className = 'token';
      span.style.backgroundColor = colorFor(t.lp, lpMin, lpMax);
      span.title = `lp=${{Number(t.lp).toFixed(4)}}`;
      span.textContent = t.text;
      assistant.appendChild(span);
    }}
    card.appendChild(assistant);

    container.appendChild(card);
  }}

  const pageinfo = document.getElementById('pageinfo');
  const total = items.length;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  pageinfo.textContent = `Page ${{currentPage + 1}} / ${{totalPages}}`;
}}

function onScaleChange() {{
  const select = document.getElementById('scale');
  currentScaleKey = select.value;
  currentPage = 0;
  const count = document.getElementById('count');
  const total = (DATA.itemsByScale[currentScaleKey] || []).length;
  count.textContent = `— ${{total}} items (top 100)`;
  renderList();
}}

function init() {{
  const scaleSelect = document.getElementById('scale');
  for (const key of DATA.scales) {{
    const opt = document.createElement('option');
    opt.value = key; opt.textContent = key;
    scaleSelect.appendChild(opt);
  }}
  if (DATA.scales.length > 0) {{
    scaleSelect.value = DATA.scales[0];
    currentScaleKey = DATA.scales[0];
  }}
  scaleSelect.addEventListener('change', onScaleChange);

  document.getElementById('prev').addEventListener('click', () => {{
    if (currentPage > 0) {{ currentPage--; renderList(); }}
  }});
  document.getElementById('next').addEventListener('click', () => {{
    const total = (DATA.itemsByScale[currentScaleKey] || []).length;
    const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    if (currentPage < totalPages - 1) {{ currentPage++; renderList(); }}
  }});

  onScaleChange();
}}

init();
</script>
</body>
</html>
"""


def build_html(run_dir: str, cfg: RunConfig, top_n: int = 100) -> str:
    eval_path = os.path.join(run_dir, "eval_logprobs.jsonl")
    if not os.path.exists(eval_path):
        raise FileNotFoundError("eval_logprobs.jsonl not found; run evaluation first")

    by_scale = _collect_by_scale(eval_path)
    tok = _load_tokenizer(cfg)

    items_by_scale: Dict[str, List[Dict[str, Any]]] = {}
    for scale_key, recs in by_scale.items():
        items = _prepare_items_for_scale(recs, run_dir, tok, top_n)
        items_by_scale[scale_key] = items

    data = {
        "scales": list(items_by_scale.keys()),
        "itemsByScale": items_by_scale,
    }

    html = _render_html(data)
    out_path = os.path.join(run_dir, "viz.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--top_n", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = build_html(args.run_dir, cfg, args.top_n)
    print(out)


if __name__ == "__main__":
    main()
