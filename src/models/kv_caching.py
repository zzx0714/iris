"""
KV-Cache for efficient autoregressive inference in the Transformer.
Adapted from the original iris kv_caching module.
"""
from typing import Tuple, Optional
import torch


class Cache:
    def __init__(
        self,
        num_samples: int,
        num_heads: int,
        max_tokens: int,
        embed_dim: int,
        device: torch.device,
    ) -> None:
        assert embed_dim % num_heads == 0
        self._n = num_samples
        self._device = device
        self._head_dim = embed_dim // num_heads
        self._cache: Optional[torch.Tensor] = None
        self._size = 0
        self._max_tokens = max_tokens
        self._num_heads = num_heads
        self._embed_dim = embed_dim

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self._n, self._num_heads, self._size, self._head_dim

    def reset(self) -> None:
        self._cache = torch.empty(
            self._n, self._num_heads, self._max_tokens, self._head_dim,
            device=self._device,
        )
        self._size = 0

    def get(self) -> torch.Tensor:
        return self._cache[:, :, : self._size, :]

    def update(self, x: torch.Tensor) -> None:
        assert x.ndim == self._cache.ndim
        assert x.shape == (self._n, self._num_heads, x.size(2), self._head_dim)
        assert self._size + x.size(2) <= self._cache.shape[2]
        self._cache[:, :, self._size : self._size + x.size(2), :] = x
        self._size += x.size(2)

    def prune(self, mask: torch.Tensor) -> None:
        """Keep only the samples where mask[i] == True."""
        self._cache = self._cache[mask]
        self._n = self._cache.shape[0]


class KVCache:
    def __init__(
        self,
        num_samples: int,
        num_heads: int,
        max_tokens: int,
        embed_dim: int,
        device: torch.device,
    ) -> None:
        self._k_cache = Cache(num_samples, num_heads, max_tokens, embed_dim, device)
        self._v_cache = Cache(num_samples, num_heads, max_tokens, embed_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self._k_cache.shape

    def reset(self) -> None:
        self._k_cache.reset()
        self._v_cache.reset()

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._k_cache.get(), self._v_cache.get()

    def update(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self._k_cache.update(k)
        self._v_cache.update(v)

    def prune(self, mask: torch.Tensor) -> None:
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)


class KeysValues:
    """Container holding one KVCache per transformer layer."""

    def __init__(
        self,
        num_samples: int,
        num_heads: int,
        max_tokens: int,
        embed_dim: int,
        num_layers: int,
        device: torch.device,
    ) -> None:
        self._kvs = tuple(
            KVCache(num_samples, num_heads, max_tokens, embed_dim, device)
            for _ in range(num_layers)
        )

    def __getitem__(self, key: int) -> KVCache:
        return self._kvs[key]

    def __len__(self) -> int:
        return len(self._kvs)

    @property
    def size(self) -> int:
        return self._kvs[0]._k_cache._size

    def reset(self) -> None:
        for kv in self._kvs:
            kv.reset()

    def prune(self, mask: torch.Tensor) -> None:
        for kv in self._kvs:
            kv.prune(mask)


class AssignWithoutInplaceCheck(torch.autograd.Function):
    """In-place slice assignment that also works with autograd."""

    @staticmethod
    def get_slice(dim: int, start: int, stop: int) -> Tuple[slice, ...]:
        return tuple([slice(None), ] * dim + [slice(start, stop), ])

    @staticmethod
    def forward(
        ctx, input_: torch.Tensor, value: torch.Tensor, dim: int, start: int, stop: int
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        input_.data[AssignWithoutInplaceCheck.get_slice(dim, start, stop)] = value
        return input_

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:
        return (
            grad_out,
            grad_out[AssignWithoutInplaceCheck.get_slice(ctx.dim, ctx.start, ctx.stop)],
            None, None, None,
        )
