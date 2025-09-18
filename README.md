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
- `noises/noise_XXXX.npy` (base unit vectors; optionally covariance-shaped)
- `generations/gens_noise_XXXX.jsonl` (records include `noise_scale`)

## Evaluate log-probs

```bash
python -m noising.eval --config configs/example.yaml --run_dir noising/outputs/<timestamp>
```

Produces `eval_logprobs.jsonl` with per-response totals and per-token log-probs. Records include `noise_scale` for ranking.

## Visualize

Build HTML viewer:
```bash
python -m noising.viz --config configs/example.yaml --run_dir noising/outputs/<timestamp> --top_n 100
```
Open `viz.html` in your browser to explore by noise scale and page through examples with token coloring by logprob.

## Covariance estimation (optional)

Estimate residual covariance at layer `layer_index` using 10 batches (size = `batch_size`) of prompt-only inputs, save `cov.npy` and its matrix square root `cov_sqrt.npy`:

```bash
python -m noising.covariance --config configs/example.yaml
```

Noise shaping: if a `cov_sqrt.npy` is found under `noising/outputs/cov/<timestamp>/` (latest) or directly as `noising/outputs/cov_sqrt.npy`, generation will shape each random base vector `u` as:

1) Sample `u ~ N(0,I)`, normalize to unit length
2) Transform `v = cov_sqrt @ u`, then re-normalize to unit length
3) Scale by each `noise_scale`

This makes noise directions reflect the empirical covariance structure of the layer residuals.
