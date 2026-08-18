from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Not an AiC baseline checkpoint: {path}")
    return checkpoint


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        import copy

        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1 - torch.exp(torch.tensor(-self.updates / 2000.0)).item())
        model_state = model.state_dict()
        for key, value in self.ema.state_dict().items():
            source = model_state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source, alpha=1 - decay)
            else:
                value.copy_(source)

