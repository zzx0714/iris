"""
LPIPS: Learned Perceptual Image Patch Similarity.
Adapted from https://github.com/CompVis/taming-transformers.
Downloads pretrained VGG weights automatically.
"""
from collections import namedtuple
import hashlib
import os
from pathlib import Path

import requests
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16 as _vgg16_impl, VGG16_Weights
from tqdm import tqdm


# ----------------------------------------------------------------------
# Download utilities
# ----------------------------------------------------------------------
URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}
CKPT_MAP = {"vgg_lpips": "vgg.pth"}
MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def _download(url: str, local_path: str, chunk_size: int = 1024) -> None:
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total = int(r.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(local_path)) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))


def _md5_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _get_ckpt_path(name: str) -> str:
    root = Path.home() / ".cache/iris/tokenizer_pretrained_vgg"
    path = root / CKPT_MAP[name]
    if not os.path.exists(path):
        print(f"Downloading {name} from {URL_MAP[name]} ...")
        _download(URL_MAP[name], str(path))
        assert _md5_hash(str(path)) == MD5_MAP[name]
    return str(path)


# ----------------------------------------------------------------------
# VGG feature extractor
# ----------------------------------------------------------------------
class vgg16(torch.nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg_pretrained = _vgg16_impl(weights=weights).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):       self.slice1.add_module(str(x), vgg_pretrained[x])
        for x in range(4, 9):     self.slice2.add_module(str(x), vgg_pretrained[x])
        for x in range(9, 16):    self.slice3.add_module(str(x), vgg_pretrained[x])
        for x in range(16, 23):   self.slice4.add_module(str(x), vgg_pretrained[x])
        for x in range(23, 30):   self.slice5.add_module(str(x), vgg_pretrained[x])
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        h = self.slice1(x); h_relu1_2 = h
        h = self.slice2(h); h_relu2_2 = h
        h = self.slice3(h); h_relu3_3 = h
        h = self.slice4(h); h_relu4_3 = h
        h = self.slice5(h); h_relu5_3 = h
        VggOutputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return VggOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


# ----------------------------------------------------------------------
# Helper layers
# ----------------------------------------------------------------------
class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """1×1 conv as a "linear" layer on top of features."""
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()
        layers = [nn.Dropout(), ] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ----------------------------------------------------------------------
# LPIPS main class
# ----------------------------------------------------------------------
def _normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def _spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


class LPIPS(nn.Module):
    def __init__(self, use_dropout: bool = True) -> None:
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lins = nn.ModuleList([
            NetLinLayer(c, 1, use_dropout) for c in self.chns
        ])
        self.load_from_pretrained()
        for p in self.parameters():
            p.requires_grad = False

    def load_from_pretrained(self) -> None:
        ckpt = _get_ckpt_path("vgg_lpips")
        self.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns scalar LPIPS distance."""
        in0 = self.scaling_layer(input)
        in1 = self.scaling_layer(target)
        outs0 = self.net(in0)
        outs1 = self.net(in1)
        diffs = {}
        for kk in range(len(self.chns)):
            f0 = _normalize_tensor(outs0[kk])
            f1 = _normalize_tensor(outs1[kk])
            diffs[kk] = (f0 - f1) ** 2
        res = [_spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for i in range(1, len(res)):
            val = val + res[i]
        return val
