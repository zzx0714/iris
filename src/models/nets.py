"""
VQ-VAE Encoder & Decoder (VQGAN-style with ResNet blocks and attention).
Adapted from https://github.com/CompVis/taming-transformers.
"""
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn


@dataclass
class EncoderDecoderConfig:
    resolution: int
    in_channels: int
    z_channels: int
    ch: int
    ch_mult: List[int]
    num_res_blocks: int
    attn_resolutions: List[int]
    out_ch: int
    dropout: float = 0.0


# ----------------------------------------------------------------------
# Normalization & nonlinearities
# ----------------------------------------------------------------------
def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish activation."""
    return x * torch.sigmoid(x)


class Normalize(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


# ----------------------------------------------------------------------
# Spatial resampling
# ----------------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = True) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = True) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


# ----------------------------------------------------------------------
# Residual block
# ----------------------------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = Normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


# ----------------------------------------------------------------------
# Self-attention block over spatial dimensions
# ----------------------------------------------------------------------
class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)   # (b, hw, c)
        k = k.reshape(b, c, h * w)                     # (b, c, hw)
        w_ = torch.bmm(q, k) * (int(c) ** -0.5)      # (b, hw, hw)
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)                      # (b, hw, hw)
        h_ = torch.bmm(v, w_)                        # (b, c, hw)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


# ----------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.num_resolutions = len(config.ch_mult)
        temb_ch = 0

        self.conv_in = nn.Conv2d(config.in_channels, config.ch, kernel_size=3, stride=1, padding=1)
        curr_res = config.resolution

        self.down = nn.ModuleList()
        in_ch_mult = (1,) + tuple(config.ch_mult)
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(config.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in, out_channels=block_out,
                                temb_channels=temb_ch, dropout=config.dropout)
                )
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down_block = nn.Module()
            down_block.block = block
            down_block.attn = attn
            if i_level != self.num_resolutions - 1:
                down_block.downsample = Downsample(block_in, with_conv=True)
                curr_res = curr_res // 2
            self.down.append(down_block)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                        temb_channels=temb_ch, dropout=config.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                        temb_channels=temb_ch, dropout=config.dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, config.z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, z_channels, h, w) latent."""
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.config.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# ----------------------------------------------------------------------
# Decoder
# ----------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.num_resolutions = len(config.ch_mult)
        temb_ch = 0

        in_ch_mult = (1,) + tuple(config.ch_mult)
        block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv2d(config.z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                        temb_channels=temb_ch, dropout=config.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                        temb_channels=temb_ch, dropout=config.dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(config.num_res_blocks + 1):
                block.append(
                    ResnetBlock(in_channels=block_in, out_channels=block_out,
                                temb_channels=temb_ch, dropout=config.dropout)
                )
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up_block = nn.Module()
            up_block.block = block
            up_block.attn = attn
            if i_level != 0:
                up_block.upsample = Upsample(block_in, with_conv=True)
                curr_res = curr_res * 2
            self.up.insert(0, up_block)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, config.out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs (B, out_ch, H, W) image from latent."""
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.config.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
