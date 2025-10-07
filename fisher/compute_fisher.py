"""
Fisher Information Matrix Computation Pipeline

Computes Fisher Information Matrices from language model outputs on a dataset,
with SVD compression for efficient storage.
"""

import torch
import yaml
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_path: str, hook_layer_name: str, device: str = "cuda"):
    """
    Load model in bfloat16 precision and tokenizer.
    Create a learnable bias parameter and register a hook to add it to the residual stream.

    Args:
        model_path: HuggingFace model path
        hook_layer_name: Name of the layer to hook (e.g., "model.layers.15")
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, bias_param)
    """
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set padding token if not already set (needed for left padding)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # Disable gradients for all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Find the layer to hook
    target_layer = None
    for name, module in model.named_modules():
        if name == hook_layer_name:
            target_layer = module
            print(f"Found target layer: {name}")
            break

    if target_layer is None:
        raise ValueError(f"Could not find layer: {hook_layer_name}")

    # Determine hidden dimension from the model config
    hidden_dim = model.config.hidden_size
    print(f"Hidden dimension: {hidden_dim}")

    # Create a learnable bias parameter initialized to zero
    bias_param = torch.nn.Parameter(
        torch.zeros(hidden_dim, dtype=torch.bfloat16, device=device)
    )
    bias_param.requires_grad = True
    print(f"Created bias parameter with shape: {bias_param.shape}")

    # Register forward hook to add bias to residual stream
    def add_bias_hook(module, input, output):
        """Hook function that adds bias to the layer output."""
        # output is typically a tuple (hidden_states, ...) for transformer layers
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Add bias to all positions in the sequence
            modified_hidden_states = hidden_states + bias_param.view(1, 1, -1)
            return (modified_hidden_states,) + output[1:]
        else:
            # If output is just the hidden states tensor
            return output + bias_param.view(1, 1, -1)

    target_layer.register_forward_hook(add_bias_hook)
    print(f"Registered forward hook on {hook_layer_name}")

    model.eval()

    return model, tokenizer, bias_param


def load_prompts(dataset_name: str, num_prompts: int) -> List[str]:
    """
    Load prompts from HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name
        num_prompts: Number of prompts to load

    Returns:
        List of prompt strings
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    prompts = []
    for i, item in enumerate(dataset):
        if i >= num_prompts:
            break
        # Extract user prompt from conversation
        prompt = item["conversation"][0]["content"]
        prompts.append(prompt)

    print(f"Loaded {len(prompts)} prompts")
    return prompts


def prepare_batch(
    prompts: List[str],
    tokenizer,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of prompts with left padding.

    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer instance
        device: Device to place tensors on

    Returns:
        Tuple of (input_ids, attention_mask)
    """
    # Apply chat template to each prompt
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in prompts
    ]

    # Tokenize with left padding
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        formatted_prompts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )

    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def get_final_token_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Run forward pass and extract logits for the final token of each sequence.
    Gradients are enabled for the forward pass.

    Args:
        model: Language model
        input_ids: Input token IDs (batch_size, seq_len)
        attention_mask: Attention mask (batch_size, seq_len)

    Returns:
        Final token logits (batch_size, vocab_size)
    """
    # Forward pass with gradients enabled
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Get the position of the last non-padded token for each sequence
    sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)

    # Extract logits for the final token of each sequence
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    final_logits = logits[batch_indices, sequence_lengths]  # (batch_size, vocab_size)

    return final_logits


def compute_fisher_matrix_vectorized(
    logits: torch.Tensor,
    bias_param: torch.nn.Parameter,
    top_k: int = 10
) -> torch.Tensor:
    """
    Compute Fisher Information Matrix with respect to MLP bias parameter.

    The Fisher Information Matrix is computed as:
    F_ij = sum_t p_t × ∂log(p_t)/∂θ_i × ∂log(p_t)/∂θ_j

    where:
    - θ is the MLP bias vector
    - p_t is the probability of token t
    - t indexes only the top-k highest probability tokens

    Args:
        logits: Logits for a single prompt (vocab_size,)
        bias_param: The MLP bias parameter (bias_dim,)
        top_k: Number of top probability tokens to use (default: 10)

    Returns:
        Fisher Information Matrix (bias_dim, bias_dim)
    """
    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (vocab_size,)
    probs = torch.exp(log_probs)  # (vocab_size,)

    bias_dim = bias_param.shape[0]

    # Get top-k tokens by probability
    top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)

    # Compute gradients only for top-k tokens
    gradients = []

    for i in range(top_k):
        # Zero gradients
        if bias_param.grad is not None:
            bias_param.grad.zero_()

        # Get the token index
        token_idx = top_k_indices[i]

        # Compute gradient of log(p_t) with respect to bias
        log_probs[token_idx].backward(retain_graph=True)

        # Store gradient
        gradients.append(bias_param.grad.clone().to(torch.float32))

    # Stack all gradients: (top_k, bias_dim)
    gradients = torch.stack(gradients)

    # Compute Fisher matrix: F = sum_t p_t × (g_t ⊗ g_t)
    # This is equivalent to: F = G^T @ diag(top_k_probs) @ G
    # where G is the gradient matrix (top_k, bias_dim)
    probs_expanded = top_k_probs.unsqueeze(1).to(torch.float32)  # (top_k, 1)
    weighted_gradients = gradients * probs_expanded  # (top_k, bias_dim)
    fisher_matrix = weighted_gradients.t() @ gradients  # (bias_dim, bias_dim)

    return fisher_matrix


def compress_with_svd(
    fisher_matrix: torch.Tensor,
    k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compress Fisher matrix using SVD, keeping top-k singular values.

    Args:
        fisher_matrix: Fisher Information Matrix (vocab_size, vocab_size)
        k: Number of singular values to keep

    Returns:
        Tuple of (U, S, V) where:
        - U: Left singular vectors (vocab_size, k)
        - S: Top k singular values (k,)
        - V: Right singular vectors (k, vocab_size)
    """
    # Detach from computation graph and convert to float32 for SVD (stay on GPU)
    F_gpu = fisher_matrix.detach().float()

    # Compute SVD on GPU
    U, S, Vh = torch.linalg.svd(F_gpu, full_matrices=False)

    # Keep top k components
    U_k = U[:, :k]  # (vocab_size, k)
    S_k = S[:k]     # (k,)
    V_k = Vh[:k, :]  # (k, vocab_size)

    # Move to CPU and convert to numpy for storage
    return U_k.cpu().numpy(), S_k.cpu().numpy(), V_k.cpu().numpy()


def save_compressed_fisher_matrices(
    compressed_matrices: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: str
):
    """
    Save compressed Fisher matrices to disk.

    Args:
        compressed_matrices: List of (U, S, V) tuples for each prompt
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as compressed numpy archive
    save_dict = {}
    for i, (U, S, V) in enumerate(compressed_matrices):
        save_dict[f"U_{i}"] = U
        save_dict[f"S_{i}"] = S
        save_dict[f"V_{i}"] = V

    output_file = output_path / "fisher_matrices_compressed.npz"
    np.savez_compressed(output_file, **save_dict)
    print(f"Saved compressed Fisher matrices to {output_file}")

    # Save metadata
    metadata = {
        "num_prompts": len(compressed_matrices),
        "num_singular_values": len(compressed_matrices[0][1]) if compressed_matrices else 0,
        "bias_dim": compressed_matrices[0][0].shape[0] if compressed_matrices else 0
    }

    metadata_file = output_path / "metadata.yaml"
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f)
    print(f"Saved metadata to {metadata_file}")


def main():
    """Main execution function."""
    # Load configuration
    config = load_config("fisher/config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    model, tokenizer, bias_param = load_model_and_tokenizer(
        config["model_path"],
        config["hook_layer_name"],
        device
    )

    # Load prompts
    prompts = load_prompts(config["dataset_name"], config["num_prompts"])

    # Process prompts in batches
    batch_size = config["batch_size"]
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    compressed_matrices = []

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        # Prepare batch with left padding
        input_ids, attention_mask = prepare_batch(batch_prompts, tokenizer, device)

        # Get final token logits (with gradients enabled)
        final_logits = get_final_token_logits(model, input_ids, attention_mask)

        # Compute Fisher matrix for each prompt in the batch
        for j in range(final_logits.size(0)):
            logits = final_logits[j]  # (vocab_size,)

            # Compute Fisher Information Matrix with respect to MLP bias
            fisher_matrix = compute_fisher_matrix_vectorized(
                logits,
                bias_param,
                top_k=config["top_k_tokens"]
            )

            # Compress with SVD
            U, S, V = compress_with_svd(fisher_matrix, config["num_singular_values"])
            compressed_matrices.append((U, S, V))

    # Save results
    save_compressed_fisher_matrices(compressed_matrices, config["output_dir"])

    print(f"\nCompleted! Processed {len(compressed_matrices)} prompts.")
    print(f"Each Fisher matrix compressed from ({fisher_matrix.shape}) to "
          f"U({U.shape}), S({S.shape}, {V.shape})")


if __name__ == "__main__":
    main()
