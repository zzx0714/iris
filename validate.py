"""
Validation script: Retrieval evaluation on the validation set.
Retrieval protocol:
  For each episode:
    1. Encode all frames → discrete tokens Z_x (Tokenizer).
    2. For each query t (t=0..T-1-stride):
         Context: frames 0..t
         Action:  batch.actions[:t+1]
         Predict: frame t+stride via WorldModel
         Query: predicted tokens Z^x_t = argmax(logits_observations at goal position)
    3. Key pool: all real tokens Z_x in the episode.
    4. For each query, compute CE(predicted_token, key_token) for ALL keys.
    5. Sort by CE, compute Hit@1/5/10.
  Retrieval is episode-internal only.

Output format matches l1_results.txt:
  Ep_<id>_Q<N>    Top10=[...]    GT=...    CE=...

Usage:
    python validate.py \
        --checkpoint experiments/run_001/checkpoint_epoch_050.pth \
        --exp_dir experiments/run_001 \
        --img_size 64 --batch_size 1 --stride 5 --seq_len 5 \
        --device cuda --output l1_results.txt
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    Tokenizer, WorldModel,
    Encoder, Decoder, EncoderDecoderConfig, TransformerConfig,
)
from src.data import DroidDataset
from src.batch import Batch
from retrieval import run_retrieval, RetrievalConfig


class EvalConfig:
    data_root    = "/data3/v-jepa/jepa-wms/dataset/droid_lerobot_dataset"
    splits_path  = "/data3/v-jepa/jepa-wms/1w2splits.json"
    img_size     = 256
    stride       = 5
    seq_len      = 5
    batch_size   = 1       # one episode per batch
    num_workers  = 4
    # Tokenizer (must match train config)
    vocab_size   = 1024
    embed_dim    = 512
    ch           = 128
    ch_mult      = [1, 1, 1, 1, 1]
    num_res_blocks = 2
    attn_resolutions = [16]
    # WorldModel (must match train config)
    wm_num_layers = 10
    wm_num_heads  = 8
    wm_embed_dim  = 512
    wm_max_blocks = 20
    act_dim       = 7
    device       = "cuda"


def build_models(cfg: EvalConfig, device):
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
        tokens_per_block = K + 1,       # K obs + 1 action
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
        obs_vocab_size = cfg.vocab_size,
        config         = tcfg,
        act_dim        = cfg.act_dim,
    ).to(device)

    return tokenizer, worldmodel, K


def _to_device(batch: Batch, device):
    def _t(x):
        return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
    return Batch(
        observations  = _t(batch.observations),
        actions       = _t(batch.actions),
        states        = _t(batch.states),
        mask_padding  = _t(batch.mask_padding),
        episode_idx   = batch.episode_idx,
        frame_idx     = _t(batch.frame_idx) if batch.frame_idx is not None else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--exp_dir",     type=str, default="./experiments/run_001")
    parser.add_argument("--img_size",    type=int, default=EvalConfig.img_size)
    parser.add_argument("--batch_size",  type=int, default=EvalConfig.batch_size)
    parser.add_argument("--stride",      type=int, default=EvalConfig.stride)
    parser.add_argument("--seq_len",     type=int, default=EvalConfig.seq_len)
    parser.add_argument("--vocab_size",  type=int, default=EvalConfig.vocab_size)
    parser.add_argument("--embed_dim",   type=int, default=EvalConfig.embed_dim)
    parser.add_argument("--device",       type=str, default=EvalConfig.device)
    parser.add_argument("--output",       type=str, default="l1_results.txt")
    parser.add_argument("--num_workers",type=int, default=EvalConfig.num_workers)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = EvalConfig()
    for k in ["img_size", "batch_size", "stride", "seq_len", "vocab_size", "embed_dim"]:
        setattr(cfg, k, getattr(args, k))

    output_path = Path(args.exp_dir) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, worldmodel, K = build_models(cfg, device)

    print(f"Loading checkpoint: {args.checkpoint}")
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
    )
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
