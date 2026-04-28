"""
Autoregressive World Model built on top of the Transformer.
Adapted from the original iris world_model.py.

Token layout (tokens_per_block = K_obs + 1):
  Block for timestep t:
    [ obs_t[0], ..., obs_t[K_obs-1], action_t ]
  Flat interleaved sequence (T_total = L*(K_obs+1)):
    [ obs_0, ..., obs_{K-1}, act_0, obs_0, ..., obs_{K-1}, act_1, ... ]

  - Uses block_causal attention: within each block all tokens attend to
    each other (obs tokens can see the action in the same block).
    Between blocks attention is strictly causal.
  - WorldModel predicts the NEXT K_obs observation tokens at each step.

  Output shapes:
    logits_observations: (B, L*K_obs, vocab_size)
"""
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer, TransformerConfig
from src.utils import init_weights


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor     # (B, L*(K+1), embed_dim)
    logits_observations: torch.FloatTensor # (B, L*K, vocab_size)


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_vocab_size: int,
        config: TransformerConfig,
        act_dim: int = 7,
    ) -> None:
        super().__init__()
        self.obs_vocab_size = obs_vocab_size
        self.config = config
        self.act_dim = act_dim
        self.transformer = Transformer(config)

        # tokens_per_block = K_obs + 1 (K_obs obs tokens + 1 action token)
        self.K_obs = config.tokens_per_block - 1
        assert self.K_obs > 0, "tokens_per_block must be > 1"

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        # Embeddings: obs via lookup table, action via linear projection → single token
        self.obs_embed = nn.Embedding(obs_vocab_size, config.embed_dim)
        self.act_proj  = nn.Linear(act_dim, config.embed_dim)  # 7-dim → single embedding

        # Heads
        self.obs_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, obs_vocab_size),
        )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return f"WorldModel(obs_vocab={self.obs_vocab_size}, K_obs={self.K_obs}, act_dim={self.act_dim})"

    def _build_sequence(
        self,
        obs_tokens: torch.LongTensor,   # (B, L, K_obs)
        actions: torch.Tensor,          # (B, L, act_dim)
    ) -> torch.Tensor:
        """
        Build interleaved sequence: [obs[0..K-1], act] per block.
        block_causal mask ensures obs tokens can see the action in the same block.
        Returns: (B, L*(K+1), embed_dim)
        """
        B, L, K = obs_tokens.shape
        device = obs_tokens.device

        # obs embeddings: (B, L*K, E)
        obs_emb = self.obs_embed(obs_tokens.reshape(B, L * K))
        obs_emb = obs_emb.reshape(B, L, K, -1)           # (B, L, K, E)

        # action embedding: (B, L, E) — single token per timestep
        act_emb = self.act_proj(actions).unsqueeze(2)   # (B, L, 1, E)

        # Interleave: [obs[0..K-1], act] per block
        combined = torch.cat([obs_emb, act_emb], dim=2)  # (B, L, K+1, E)
        seq = combined.reshape(B, L * (K + 1), -1)       # (B, L*(K+1), E)
        return seq

    def _obs_indices(self, L: int, device: torch.device) -> torch.Tensor:
        """Positions of observation tokens in the flat interleaved sequence.
        Layout: [obs0, ..., obs_{K-1}, act] per block → obs at 0..K-1."""
        K = self.K_obs
        idx = []
        for i in range(L):
            for j in range(K):
                idx.append(i * (K + 1) + j)
        return torch.tensor(idx, device=device, dtype=torch.long)

    def forward(
        self,
        obs_tokens: torch.LongTensor,   # (B, L, K_obs)
        actions: torch.Tensor,          # (B, L, act_dim)
        past_keys_values = None,
    ) -> WorldModelOutput:
        B, L, K = obs_tokens.shape
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        # Build and embed sequence: (B, L*(K+1), E)
        seq = self._build_sequence(obs_tokens, actions)
        positions = prev_steps + torch.arange(seq.size(1), device=obs_tokens.device)
        seq = seq + self.pos_emb(positions)

        # Transformer
        x = self.transformer(seq, past_keys_values)

        # Select obs positions
        obs_idx = self._obs_indices(L, x.device)
        x_obs = x[:, obs_idx, :]   # (B, L*K, E)

        logits_obs = self.obs_head(x_obs)  # (B, L*K, vocab)

        return WorldModelOutput(x, logits_obs)

    def compute_loss(
        self,
        obs_tokens: torch.LongTensor,    # (B, L, K_obs)
        actions: torch.Tensor,            # (B, L, act_dim) continuous
        mask_padding: torch.Tensor,      # (B, L) bool
    ) -> tuple[torch.Tensor, dict]:
        """
        WorldModel predicts the next frame's K_obs tokens for each context block.
        Loss: predict obs[t+1] from context obs[t] + action[t].
        Labels: obs[t+1] for t = 0..L-2 (L-1 context blocks).
        """
        B, L, K = obs_tokens.shape
        outputs = self(obs_tokens, actions)

        # The model predicts K tokens for each of the L context blocks.
        # We only have labels for L-1 blocks (no action after the last frame).
        # Truncate to L-1 blocks: logits_obs[:, :(L-1)*K, :]
        logits_obs_trunc = outputs.logits_observations[:, : (L - 1) * K, :]

        # Observation labels: obs[t+1] for t = 0..L-2 → (B, (L-1)*K)
        labels_obs = obs_tokens[:, 1:, :].reshape(B, (L - 1) * K)
        pad_2d = mask_padding[:, 1:].unsqueeze(-1).expand(B, L - 1, K)
        pad_mask = pad_2d.reshape(B, (L - 1) * K)
        labels_obs = labels_obs.masked_fill(~pad_mask, -100)

        logits_obs_flat = logits_obs_trunc.reshape(-1, self.obs_vocab_size)

        loss_obs = F.cross_entropy(logits_obs_flat, labels_obs.reshape(-1))

        return loss_obs, {
            "loss_obs": loss_obs,
        }
