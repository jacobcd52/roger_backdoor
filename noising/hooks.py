from __future__ import annotations

from typing import Any, Optional

import torch

from .model import get_transformer_layers


class ResidualVectorAdder:
    def __init__(self, model: Any, layer_index: int, noise_vector: torch.Tensor) -> None:
        self.model = model
        self.layer_index = int(layer_index)
        self.noise_vector = noise_vector
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def register(self) -> None:
        layers = get_transformer_layers(self.model)
        if not (0 <= self.layer_index < len(layers)):
            raise IndexError(f"layer_index {self.layer_index} out of range [0, {len(layers)-1}]")
        target = layers[self.layer_index]
        self._handle = target.register_forward_pre_hook(self._pre_hook, with_kwargs=True)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _pre_hook(self, module: torch.nn.Module, args, kwargs):  # type: ignore[override]
        # Expected signature: (hidden_states, ...)
        if not args:
            return args, kwargs
        hidden_states = args[0]
        if hidden_states is None:
            return args, kwargs
        noise = self.noise_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        while noise.dim() < hidden_states.dim():
            noise = noise.unsqueeze(0)
        noise = noise.expand_as(hidden_states)
        new_h = hidden_states + noise
        new_args = (new_h,) + tuple(args[1:])
        return new_args, kwargs
