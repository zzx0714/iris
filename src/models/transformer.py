"""
GPT-style autoregressive Transformer.
Adapted from https://github.com/karpathy/minGPT + iris.
"""
from dataclasses import dataclass
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_caching import KeysValues


@dataclass
class TransformerConfig:
    tokens_per_block: int      # K + 1 (obs tokens + 1 action token)
    max_blocks: int           # Maximum number of blocks in context
    attention: str            # "causal" | "block_causal"
    num_layers: int
    num_heads: int
    embed_dim: int
    embed_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0

    @property
    def max_tokens(self) -> int:
        return self.tokens_per_block * self.max_blocks


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int) -> KeysValues:
        return KeysValues(
            n,
            self.config.num_heads,
            self.config.max_tokens,
            self.config.embed_dim,
            self.config.num_layers,
            self.ln_f.weight.device,
        )

    def forward(self, sequences: torch.Tensor, past: Optional[KeysValues] = None) -> torch.Tensor:
        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            x = block(x, None if past is None else past[i])
        x = self.ln_f(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, kv_cache: Optional) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), kv_cache)
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        assert config.attention in ("causal", "block_causal")
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        self.key   = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj  = nn.Linear(config.embed_dim, config.embed_dim)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # Build causal mask
        max_t = config.max_tokens
        causal = torch.tril(torch.ones(max_t, max_t))
        if config.attention == "block_causal":
            block = torch.ones(config.tokens_per_block, config.tokens_per_block)
            block_causal = torch.block_diag(*[block for _ in range(config.max_blocks)])
            mask = torch.max(causal, block_causal)
        else:
            mask = causal
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional) -> torch.Tensor:
        B, T, C = x.size()
        if kv_cache is not None:
            _, nh, L, _ = kv_cache.shape
            assert nh == self.num_heads and C == self.num_heads * self.head_dim
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[L : L + T, : L + T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y))
