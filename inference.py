"""
Inference: Load saved weights and run retrieval on the validation set.

Usage:
    python inference.py \
        --checkpoint experiments/run_001/checkpoint_epoch_050.pth \
        --exp_dir experiments/run_001 \
        --img_size 256 --stride 5 --seq_len 5 \
        --device cuda --output inference_results.txt \
        --max_episodes 500
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    Tokenizer, WorldModel,
    Encoder, Decoder, EncoderDecoderConfig, TransformerConfig,
)
from src.data import DroidDataset
from retrieval import run_retrieval, RetrievalConfig


class InferConfig:
    img_size     = 256
    stride       = 5
    seq_len      = 5
    batch_size   = 8
    num_workers  = 4
    vocab_size   = 1024
    embed_dim    = 512
    ch           = 128
    ch_mult      = [1, 1, 1, 1, 1]
    num_res_blocks = 2
    attn_resolutions = [16]
    wm_num_layers = 10
    wm_num_heads  = 7
    wm_embed_dim  = 448
    wm_max_blocks = 20
    act_dim       = 7
    device       = "cuda"


def build_models(cfg, device):
    enc_cfg = EncoderDecoderConfig(
        resolution=cfg.img_size, in_channels=3, z_channels=cfg.embed_dim,
        ch=cfg.ch, ch_mult=cfg.ch_mult, num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions, out_ch=3, dropout=0.0,
    )
    tokenizer = Tokenizer(
        vocab_size=cfg.vocab_size, embed_dim=cfg.embed_dim,
        encoder=Encoder(enc_cfg), decoder=Decoder(enc_cfg), with_lpips=False,
    ).to(device)

    latent_h, latent_w = tokenizer.latent_hw
    K = latent_h * latent_w

    tcfg = TransformerConfig(
        tokens_per_block = K + 1,
        max_blocks      = cfg.wm_max_blocks,
        attention       = "block_causal",
        num_layers      = cfg.wm_num_layers,
        num_heads       = cfg.wm_num_heads,
        embed_dim       = cfg.wm_embed_dim,
        embed_pdrop     = 0.1,
        resid_pdrop     = 0.1,
        attn_pdrop      = 0.1,
    )
    worldmodel = WorldModel(
        obs_vocab_size=cfg.vocab_size,
        config=tcfg,
        act_dim=cfg.act_dim,
    ).to(device)

    return tokenizer, worldmodel, K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--exp_dir",     type=str, default="./experiments/run_001")
    parser.add_argument("--img_size",    type=int, default=InferConfig.img_size)
    parser.add_argument("--stride",      type=int, default=InferConfig.stride)
    parser.add_argument("--seq_len",     type=int, default=InferConfig.seq_len)
    parser.add_argument("--batch_size",  type=int, default=InferConfig.batch_size)
    parser.add_argument("--vocab_size",  type=int, default=InferConfig.vocab_size)
    parser.add_argument("--embed_dim",   type=int, default=InferConfig.embed_dim)
    parser.add_argument("--device",      type=str, default=InferConfig.device)
    parser.add_argument("--output",      type=str, default="inference_results.txt")
    parser.add_argument("--num_workers", type=int, default=InferConfig.num_workers)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = InferConfig()
    for k in ["img_size", "stride", "seq_len", "batch_size", "vocab_size", "embed_dim"]:
        setattr(cfg, k, getattr(args, k))

    output_path = Path(args.exp_dir) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, worldmodel, K = build_models(cfg, device)

    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer.load_state_dict(ckpt["tokenizer"])
    worldmodel.load_state_dict(ckpt["worldmodel"])
    print(f"Loaded. K_obs={K}")

    val_ds = DroidDataset(
        split="val", stride=cfg.stride, seq_len=cfg.seq_len,
        img_size=cfg.img_size, preload=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_ds.collate_fn, pin_memory=True,
    )
    print(f"Val episodes: {len(val_ds)}")

    metrics = run_retrieval(
        tokenizer, worldmodel, val_loader, device, cfg, K,
        output_path=str(output_path),
        max_episodes=args.max_episodes,
    )
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
