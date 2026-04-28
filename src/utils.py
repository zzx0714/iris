"""
Shared utility classes.
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class LossWithIntermediateLosses:
    commitment_loss: Any = None
    reconstruction_loss: Any = None
    perceptual_loss: Any = None
    loss_obs: Any = None
    loss_actions: Any = None
    loss_values: Any = None
    loss_entropy: Any = None

    def reduce(self):
        parts = []
        for field in ['commitment_loss', 'reconstruction_loss', 'perceptual_loss',
                       'loss_obs',
                       'loss_actions', 'loss_values', 'loss_entropy']:
            v = getattr(self, field)
            if v is not None:
                parts.append(v)
        return sum(parts)


def init_weights(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)


import torch
