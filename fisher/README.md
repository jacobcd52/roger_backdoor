# Fisher Information Matrix Computation Pipeline

This pipeline computes Fisher Information Matrices from language model outputs on a dataset, with SVD compression for efficient storage.

## Overview

The pipeline:
1. Loads a language model in bfloat16 precision
2. Creates a learnable bias parameter (initialized to zero) and registers a hook to add it to the residual stream
3. Processes prompts from a HuggingFace dataset in batches
4. Extracts final token logits and computes log probabilities
5. Computes Fisher Information Matrix w.r.t. the bias for each prompt (using gradients from top-k tokens)
6. Compresses each matrix using SVD, keeping top-k singular values
7. Saves compressed matrices to disk

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:

- `model_path`: HuggingFace model path (default: FabienRoger/backdoor_250913_1B)
- `dataset_name`: HuggingFace dataset (default: lmsys/lm-sys-chat-1m)
- `batch_size`: Batch size for processing (default: 8)
- `num_prompts`: Number of prompts to process (default: 100)
- `hook_layer_name`: Name of the layer to hook for adding bias to residual stream (default: model.layers.15)
- `top_k_tokens`: Number of top probability tokens to use for Fisher computation (default: 10)
- `num_singular_values`: Number of singular values to keep in SVD (default: 10)
- `output_dir`: Directory for saving results (default: fisher_results)

## Usage

```bash
python compute_fisher.py
```

## Output

The pipeline saves:

1. `fisher_matrices_compressed.npz`: Compressed Fisher matrices
   - For each prompt i: `U_{i}`, `S_{i}`, `V_{i}` arrays
   - Original matrix can be reconstructed: `F ≈ U @ diag(S) @ V`

2. `metadata.yaml`: Metadata about the computation
   - Number of prompts processed
   - Number of singular values kept
   - Bias dimension (hidden size of the model)

## Fisher Information Matrix Computation

For each prompt, the Fisher Information Matrix is computed with respect to a bias vector added to the residual stream:

```
F_ij = sum_t p_t × ∂log(p_t)/∂θ_i × ∂log(p_t)/∂θ_j
```

where:
- `θ` is a learnable bias vector (initialized to zero) added to the residual stream at a specified layer
- `p_t` is the probability of token t in the vocabulary
- `t` indexes only the top-k highest probability tokens (configurable via `top_k_tokens`)

The computation:
1. Creates a learnable bias parameter (initialized to zero) with dimension equal to the hidden size
2. Registers a forward hook on the specified layer (configurable via `hook_layer_name`) that adds this bias to the residual stream
3. Runs forward pass with this bias (set to zero initially)
4. Identifies the top-k tokens with highest probability
5. For each of the k tokens, computes `∂log(p_t)/∂θ` via backpropagation (k separate backward passes)
6. Accumulates the weighted outer products: `F = sum_t p_t × (g_t ⊗ g_t)`
7. Uses vectorized matrix multiplication: `F = G^T @ diag(p) @ G`

This approach:
- Works with models that don't have bias parameters
- Computes Fisher w.r.t. perturbations to the residual stream at a specific layer
- Is much more efficient than looping over the entire vocabulary, as we only compute gradients for the most likely tokens

## Memory Efficiency

- Full Fisher matrices are `(hidden_dim × hidden_dim)` per prompt
  - Where `hidden_dim` is the hidden size of the model (e.g., 2048 for many LLMs)
- With SVD compression keeping k=10 singular values:
  - Storage: `U(hidden_dim × k) + S(k) + V(k × hidden_dim)`
  - Significant reduction when hidden_dim is large (e.g., 2048² → 2×2048×10 + 10)
