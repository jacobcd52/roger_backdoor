# Residual Noising Runner

This repo provides tools to:

- Load a HuggingFace language model in bf16
- Load prompts from `lmsys/lmsys-chat-1m` (first N prompts configurable)
- Format with `apply_chat_template(add_generation_prompt=True)`
- Generate responses while adding a fixed random vector to a chosen layer's residual stream
- Save all generations and associated noise vectors
- Evaluate per-token and total log-probs of the generated assistant text under a reference model
- Visualize prompts/vectors with the worst log-probs and colorize token log-probs

## Setup

```bash
pip install -U pip
pip install -r requirements.txt
```

## Config

Edit `configs/example.yaml` or create your own. Key fields:

- `model_name`: HF model repo id
- `reference_model_name`: optional; defaults to `model_name`
- `dtype`: `bf16` recommended
- `dataset_name`: `lmsys/lmsys-chat-1m`
- `num_prompts`: e.g., 1000
- `max_prompt_len`: filter out prompts with tokenized length > this (default 256)
- `batch_size`, `max_new_tokens`
- `layer_index`: layer L to apply residual vector
- `noise_norms`: list of scales swept per random vector (fallback to `noise_norm`)
- `noise_norm`, `num_noise_vectors`, `seed`
- `output_dir`: base directory for run outputs

## Run generation

```bash
python -m noising.generate --config configs/example.yaml
```

This creates a timestamped run directory under `noising/outputs/` with:
- `config.yaml`
- `noises/noise_XXXX.npy` (base unit vectors)
- `generations/gens_noise_XXXX.jsonl` (records include `noise_scale`)

## Evaluate log-probs

```bash
python -m noising.eval --config configs/example.yaml --run_dir noising/outputs/<timestamp>
```

Produces `eval_logprobs.jsonl` with per-response totals and per-token log-probs. Records include `noise_scale` for ranking.

## Visualize

Show worst total log-probs:

```bash
python -m noising.viz --run_dir noising/outputs/<timestamp> --top_k 10
```

Show a specific example with colored tokens:

```bash
python -m noising.viz --run_dir noising/outputs/<timestamp> --noise_id 0 --prompt_index 12
```

Notes:
- You may need sufficient GPU memory for bf16 and the chosen model.
- If your model lacks a discovered layer list, adjust `get_transformer_layers` in `noising/model.py`.
