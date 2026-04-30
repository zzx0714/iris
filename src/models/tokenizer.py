"""
Vector Quantized Variational AutoEncoder (VQ-VAE) Tokenizer.
Adapted from https://github.com/CompVis/taming-transformers.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lpips import LPIPS
from .nets import Encoder, Decoder, EncoderDecoderConfig


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor           # (..., E, h, w) continuous latent before quant
    z_quantized: torch.FloatTensor # (..., E, h, w) quantized latent (straight-through)
    tokens: torch.LongTensor       # (..., h*w) discrete codebook indices


class Tokenizer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder: Encoder,
        decoder: Decoder,
        with_lpips: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.decoder = decoder

        self.pre_quant_conv = nn.Conv2d(encoder.config.z_channels, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, decoder.config.z_channels, kernel_size=1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Initialise codebook with uniform distribution over [-1/vocab, 1/vocab]
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

        if with_lpips:
            self.lpips = LPIPS().eval()
            for p in self.lpips.parameters():
                p.requires_grad = False
        else:
            self.lpips = None

        # Cache latent spatial size for use by WorldModel
        self._compute_latent_hw()

    def __repr__(self) -> str:
        return f"Tokenizer(vocab={self.vocab_size}, embed={self.embedding.weight.shape[1]})"

    # ------------------------------------------------------------------
    # Core forward: encode + decode (used during training)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:  (..., C, H, W) images in [0, 1] if preprocess, else [-1, 1]
            should_preprocess: convert [0,1]→[-1,1] before encoding
            should_postprocess: convert [-1,1]→[0,1] after decoding
        Returns:
            z, z_quantized, reconstruction
        """
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstruction = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstruction

    def encode(
        self, x: torch.Tensor, should_preprocess: bool = False
    ) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape                          # (..., C, H, W)
        x = x.view(-1, *shape[-3:])             # (B, C, H, W)
        z = self.encoder(x)                      # (B, z_channels, h, w)
        z = self.pre_quant_conv(z)              # (B, embed_dim, h, w)

        b, e, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(b * h * w, e)   # (N, embed_dim)

        # --- codebook lookup: nearest embedding by L2 distance ---
        dist = (
            (z_flat ** 2).sum(dim=1, keepdim=True)
            + (self.embedding.weight ** 2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )   # (N, vocab_size)
        tokens = dist.argmin(dim=-1)            # (N,)
        z_q = self.embedding(tokens)             # (N, embed_dim)
        z_q = z_q.reshape(b, h, w, e).permute(0, 3, 1, 2).contiguous()   # (B, E, h, w)

        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape                        # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def encode_decode(
        self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False
    ) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------
    def compute_loss(self, observations: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            observations: (B*T, C, H, W) images in [0, 1]
        Returns:
            total_loss, dict of intermediate losses
        """
        # Check for NaN/Inf in input
        if not torch.isfinite(observations).all():
            return torch.tensor(float('nan'), device=observations.device, requires_grad=True), {
                "commitment_loss": torch.tensor(float('nan'), device=observations.device),
                "recon_loss": torch.tensor(float('nan'), device=observations.device),
                "perceptual_loss": torch.tensor(float('nan'), device=observations.device),
            }

        obs_in = self.preprocess_input(observations)
        z, z_q, recon = self.forward(obs_in, should_preprocess=False, should_postprocess=False)

        # Commitment loss (beta=1.0, as in original VQ-VAE paper)
        beta = 1.0
        commitment_loss = (
            (z.detach() - z_q).pow(2).mean()
            + beta * (z - z_q.detach()).pow(2).mean()
        )

        # Reconstruction losses  (both in [-1,1] space)
        recon_loss = F.l1_loss(obs_in, recon, reduction="mean")

        # Perceptual loss (LPIPS params are frozen, but gradients flow to recon).
        # Force float32 to avoid negative values under AMP float16.
        if self.lpips:
            perc_loss = self.lpips(obs_in.float(), recon.float()).float().mean()
        else:
            perc_loss = torch.tensor(0.0, device=observations.device)

        total = commitment_loss + recon_loss + perc_loss
        return total, {
            "commitment_loss": commitment_loss,
            "recon_loss": recon_loss,
            "perceptual_loss": perc_loss,
        }

    # ------------------------------------------------------------------
    # Pre/post-processing helpers (matching original iris)
    # ------------------------------------------------------------------
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """[0, 1] → [-1, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → [0, 1]"""
        return y.add(1).div(2)

    @property
    def latent_hw(self) -> Tuple[int, int]:
        """Spatial size of one latent (h, w) after encoder. Cached at init."""
        return self._latent_h, self._latent_w

    def _compute_latent_hw(self):
        """Compute and cache latent spatial size once."""
        dummy = torch.zeros(1, 3, self.encoder.config.resolution, self.encoder.config.resolution)
        with torch.no_grad():
            z = self.encoder(dummy)
        self._latent_h, self._latent_w = z.shape[2], z.shape[3]
