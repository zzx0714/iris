"""
Token slicing utilities for selective embedding and head prediction.
Adapted from the original iris slicer module.
"""
import math
from typing import List
import torch
import torch.nn as nn


class Slicer(nn.Module):
    """Selects positions within each block based on a binary mask."""

    def __init__(self, max_blocks: int, block_mask: torch.Tensor) -> None:
        super().__init__()
        self.block_size = block_mask.size(0)
        self.num_kept = int(block_mask.sum().item())

        kept_indices = torch.where(block_mask)[0].repeat(max_blocks)
        offsets = torch.arange(max_blocks).repeat_interleave(self.num_kept)
        self.register_buffer("indices", kept_indices + block_mask.size(0) * offsets)

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> torch.Tensor:
        total = num_steps + prev_steps
        num_blocks = math.ceil(total / self.block_size)
        idx = self.indices[: num_blocks * self.num_kept]
        mask = (idx >= prev_steps) & (idx < total)
        return idx[mask] - prev_steps


class Head(Slicer):
    """A Slicer that applies a head_module to selected positions."""

    def __init__(self, max_blocks: int, block_mask: torch.Tensor, head_module: nn.Module) -> None:
        super().__init__(max_blocks, block_mask)
        self.head_module = head_module

    def forward(self, x: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        sl = self.compute_slice(num_steps, prev_steps)
        return self.head_module(x[:, sl])   # x is (B, T, E)


class Embedder(nn.Module):
    """
    Interleaved embedding of action and observation tokens.
    block_masks must partition a block (sum to 1 at each position).
    """

    def __init__(
        self,
        max_blocks: int,
        block_masks: List[torch.Tensor],
        embedding_tables: List[nn.Embedding],
    ) -> None:
        super().__init__()
        assert len(block_masks) == len(embedding_tables)
        assert all((sum(m) == 1).item() for m in block_masks)
        self.embedding_dim = embedding_tables[0].embedding_dim
        assert all(e.embedding_dim == self.embedding_dim for e in embedding_tables)
        self.embedding_tables = embedding_tables
        self.slicers = [Slicer(max_blocks, m) for m in block_masks]

    def forward(self, tokens: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        """
        Args:
            tokens: (B, T) interleaved action + observation tokens
            num_steps: length of current step
            prev_steps: length of cached context
        Returns:
            (B, T, embed_dim)
        """
        assert tokens.ndim == 2
        out = torch.zeros(
            tokens.size(0), num_steps + prev_steps, self.embedding_dim,
            device=tokens.device, dtype=tokens.dtype,
        )
        for slicer, emb in zip(self.slicers, self.embedding_tables):
            sl = slicer.compute_slice(num_steps, prev_steps)
            out[:, sl] = emb(tokens[:, sl])
        return out
