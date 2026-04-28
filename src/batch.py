"""
Batch dataclass used across the original iris codebase.
"""
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class Batch:
    observations: torch.Tensor       # (B, T, 3, H, W)  in [0, 1]
    actions: torch.Tensor           # (B, T, act_dim)
    states: torch.Tensor            # (B, T, state_dim)
    mask_padding: torch.Tensor      # (B, T)  bool, True = valid frame
    episode_idx: torch.Tensor       # (B,)    int
    frame_idx: torch.Tensor        # (B,)    int per-frame index within episode
