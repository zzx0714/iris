"""
Separate training loop for Tokenizer and WorldModel.

Stage 1: Train Tokenizer only (freeze worldmodel)
Stage 2: Train WorldModel only (freeze tokenizer)

Usage:
    # Stage 1: Train Tokenizer
    python train.py --stage tokenizer --exp_dir ./experiments/run_001

    # Stage 2: Train WorldModel
    python train.py --stage worldmodel --exp_dir ./experiments/run_001 \
        --tokenizer_ckpt ./experiments/run_001/tokenizer_final.pth
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    Tokenizer,
    WorldModel,
    Encoder,
    Decoder,
    EncoderDecoderConfig,
    TransformerConfig,
)
from src.data import DroidDataset, build_dataloaders
from src.batch import Batch


class TrainConfig:
    data_root    = "/data3/v-jepa/jepa-wms/dataset/droid_lerobot_dataset"
    splits_path  = "/data3/v-jepa/jepa-wms/1w2splits.json"
    stride       = 5
    seq_len      = 5
    img_size     = 256

    # Tokenizer
    vocab_size   = 1024
    embed_dim    = 512
    ch           = 128
    ch_mult      = [1, 1, 1, 1, 1]
    num_res_blocks = 2
    attn_resolutions = [16]

    # WorldModel (aligned with official IRIS)
    wm_num_layers  = 10
    wm_num_heads   = 7
    wm_embed_dim   = 448
    wm_embed_pdrop = 0.1
    wm_resid_pdrop = 0.1
    wm_attn_pdrop  = 0.1
    act_dim        = 7
    wm_max_blocks  = 20   # enough for retrieval (official IRIS = 20)

    # Training
    exp_dir      = "./experiments/run_001"
    batch_size   = 8

    # Tokenizer training
    tok_epochs   = 50
    lr_tokenizer = 3e-5

    # WorldModel training
    wm_epochs    = 50
    lr_worldmodel = 1e-4

    weight_decay = 0.01
    grad_clip    = 10.0
    num_workers  = 2
    save_every   = 5
    val_batches  = 50

    device       = "cuda"


def build_tokenizer(cfg: TrainConfig) -> Tokenizer:
    enc_cfg = EncoderDecoderConfig(
        resolution     = cfg.img_size,
        in_channels    = 3,
        z_channels     = cfg.embed_dim,
        ch             = cfg.ch,
        ch_mult        = cfg.ch_mult,
        num_res_blocks = cfg.num_res_blocks,
        attn_resolutions = cfg.attn_resolutions,
        out_ch         = 3,
        dropout        = 0.0,
    )
    encoder = Encoder(enc_cfg)
    decoder = Decoder(enc_cfg)
    tok = Tokenizer(
        vocab_size  = cfg.vocab_size,
        embed_dim   = cfg.embed_dim,
        encoder     = encoder,
        decoder     = decoder,
        with_lpips  = True,
    )
    print(f"[Tokenizer] latent shape: {tok.latent_hw}")
    return tok


def build_worldmodel(cfg: TrainConfig, K: int) -> WorldModel:
    # action is projected to a SINGLE token via nn.Linear(act_dim, embed_dim)
    tpb = K + 1                    # tokens per block = K_obs + 1 action token

    tcfg = TransformerConfig(
        tokens_per_block = tpb,
        max_blocks      = cfg.wm_max_blocks,
        attention       = "block_causal",
        num_layers      = cfg.wm_num_layers,
        num_heads       = cfg.wm_num_heads,
        embed_dim       = cfg.wm_embed_dim,
        embed_pdrop     = cfg.wm_embed_pdrop,
        resid_pdrop     = cfg.wm_resid_pdrop,
        attn_pdrop      = cfg.wm_attn_pdrop,
    )
    print(f"[WorldModel] tokens_per_block={tpb}, max_blocks={tcfg.max_blocks}, "
          f"K_obs={K}, act_dim={cfg.act_dim}")

    return WorldModel(
        obs_vocab_size = cfg.vocab_size,
        config         = tcfg,
        act_dim        = cfg.act_dim,
    )


def to_device(batch: Batch, device):
    def move(x):
        return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
    return Batch(
        observations  = move(batch.observations),
        actions      = move(batch.actions),
        states       = move(batch.states),
        mask_padding = move(batch.mask_padding),
        episode_idx  = batch.episode_idx,
        frame_idx    = move(batch.frame_idx) if batch.frame_idx is not None else None,
    )


def train_tokenizer(tokenizer, train_loader, cfg, device, exp_dir):
    """Stage 1: Train Tokenizer only."""
    print("\n" + "="*60)
    print("STAGE 1: TRAINING TOKENIZER")
    print("="*60)

    writer = SummaryWriter(log_dir=str(exp_dir / "logs_tok"))

    tokenizer = tokenizer.to(device)
    opt = optim.AdamW(tokenizer.parameters(), lr=cfg.lr_tokenizer, weight_decay=cfg.weight_decay)
    scaler = GradScaler('cuda')

    n_params = sum(p.numel() for p in tokenizer.parameters())
    print(f"[Tokenizer] params: {n_params/1e6:.1f}M")

    for epoch in range(cfg.tok_epochs):
        tokenizer.train()
        pbar = tqdm(train_loader, desc=f"Tok Epoch {epoch+1}/{cfg.tok_epochs}")

        for batch in pbar:
            batch = to_device(batch, device)
            obs = batch.observations   # (B, T, 3, H, W)
            B, T, C, H, W = obs.shape
            # Train tokenizer with one random frame per sequence so effective batch is B.
            t_idx = torch.randint(T, (1,), device=obs.device).item()
            obs_flat = obs[:, t_idx].float() / 255.0

            opt.zero_grad(set_to_none=True)

            with autocast('cuda'):
                tok_loss, tok_metrics = tokenizer.compute_loss(obs_flat)

            # Check for NaN
            if torch.isnan(tok_loss):
                print(f"\n[Warning] NaN loss at step {step}, skipping batch")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(tok_loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(tokenizer.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix({"loss": f"{tok_loss.item():.4f}"})

            # Logging
            step = epoch * len(train_loader) + pbar.n
            if step % 100 == 0:
                writer.add_scalar("train/loss", tok_loss.item(), step)
                for k, v in tok_metrics.items():
                    writer.add_scalar(f"train/{k}", v.item(), step)

        # Save checkpoint
        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.tok_epochs - 1:
            ckpt_path = exp_dir / f"tokenizer_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch,
                "tokenizer": tokenizer.state_dict(),
                "opt": opt.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_") and not callable(getattr(cfg, k))},
            }, ckpt_path)
            print(f"[Saved] {ckpt_path}")

    # Save final tokenizer
    final_path = exp_dir / "tokenizer_final.pth"
    torch.save({
        "epoch": cfg.tok_epochs - 1,
        "tokenizer": tokenizer.state_dict(),
        "config": {k: getattr(cfg, k) for k in dir(cfg)
                   if not k.startswith("_") and not callable(getattr(cfg, k))},
    }, final_path)
    print(f"[Saved] {final_path}")

    writer.close()
    return tokenizer


def train_worldmodel(tokenizer, train_loader, val_loader, cfg, device, exp_dir,
                     resume_path=None):
    """Stage 2: Train WorldModel only (tokenizer frozen)."""
    print("\n" + "="*60)
    print("STAGE 2: TRAINING WORLDMODEL (TOKENIZER FROZEN)")
    print("="*60)

    # Freeze tokenizer
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # Get K_obs from tokenizer
    latent_h, latent_w = tokenizer.latent_hw
    K_obs = latent_h * latent_w
    print(f"[WorldModel] K_obs = {K_obs}")

    # Build worldmodel
    worldmodel = build_worldmodel(cfg, K_obs).to(device)

    writer = SummaryWriter(log_dir=str(exp_dir / "logs_wm"))

    opt = optim.AdamW(worldmodel.parameters(), lr=cfg.lr_worldmodel, weight_decay=cfg.weight_decay)
    scaler = GradScaler('cuda')

    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_path:
        print(f"[Resuming WorldModel] from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        worldmodel.load_state_dict(ckpt["worldmodel"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
            print(f"  Restored optimizer state")
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            print(f"  Resuming from epoch {start_epoch}")

    n_params = sum(p.numel() for p in worldmodel.parameters())
    print(f"[WorldModel] params: {n_params/1e6:.1f}M")

    for epoch in range(start_epoch, cfg.wm_epochs):
        worldmodel.train()
        pbar = tqdm(train_loader, desc=f"WM Epoch {epoch+1}/{cfg.wm_epochs}")

        for batch in pbar:
            batch = to_device(batch, device)
            obs = batch.observations   # (B, T, 3, H, W)
            B, T, C, H, W = obs.shape
            obs_flat = obs.reshape(B * T, C, H, W).float() / 255.0

            # Encode with frozen tokenizer (no gradient)
            with torch.no_grad():
                enc_out = tokenizer.encode(
                    tokenizer.preprocess_input(obs_flat),
                    should_preprocess=False,
                )
                obs_tokens = enc_out.tokens.reshape(B, T, -1)   # (B, T, K)

            # WorldModel forward (gradient only here)
            opt.zero_grad(set_to_none=True)

            with autocast('cuda'):
                wm_loss, wm_metrics = worldmodel.compute_loss(
                    obs_tokens   = obs_tokens,
                    actions      = batch.actions,
                    mask_padding = batch.mask_padding,
                )

            scaler.scale(wm_loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(worldmodel.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix({"loss": f"{wm_loss.item():.4f}"})

            # Logging
            step = epoch * len(train_loader) + pbar.n
            if step % 100 == 0:
                writer.add_scalar("train/loss", wm_loss.item(), step)
                for k, v in wm_metrics.items():
                    writer.add_scalar(f"train/{k}", v.item(), step)

        # Validation every epoch
        val_metrics = validate_worldmodel(tokenizer, worldmodel, val_loader, device, cfg)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        val_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        print(f"\n[WM Epoch {epoch+1}] Val: {val_str}")

        # Save checkpoint
        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.wm_epochs - 1:
            ckpt_path = exp_dir / f"worldmodel_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch,
                "worldmodel": worldmodel.state_dict(),
                "opt": opt.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_") and not callable(getattr(cfg, k))},
            }, ckpt_path)
            print(f"[Saved] {ckpt_path}")

    # Save final worldmodel
    final_path = exp_dir / "worldmodel_final.pth"
    torch.save({
        "epoch": cfg.wm_epochs - 1,
        "worldmodel": worldmodel.state_dict(),
        "config": {k: getattr(cfg, k) for k in dir(cfg)
                   if not k.startswith("_") and not callable(getattr(cfg, k))},
    }, final_path)
    print(f"[Saved] {final_path}")

    writer.close()
    return worldmodel


def validate_worldmodel(tokenizer, worldmodel, val_loader, device, cfg, max_batches=50):
    tokenizer.eval()
    worldmodel.eval()
    total_wm = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", leave=False):
            if n >= max_batches:
                break
            batch = to_device(batch, device)
            obs = batch.observations.float() / 255.0
            B, T, C, H, W = obs.shape
            obs_flat = obs.reshape(B * T, C, H, W)

            enc_out = tokenizer.encode(
                tokenizer.preprocess_input(obs_flat), should_preprocess=False,
            )
            obs_tokens = enc_out.tokens.reshape(B, T, -1)
            wm_loss, _ = worldmodel.compute_loss(
                obs_tokens=obs_tokens,
                actions=batch.actions,
                mask_padding=batch.mask_padding,
            )
            total_wm += wm_loss.item()
            n += 1

    return {"wm_loss": total_wm / max(n, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="both",
                       choices=["tokenizer", "worldmodel", "both"],
                       help="Training stage: tokenizer, worldmodel, or both")
    parser.add_argument("--exp_dir", type=str, default="./experiments/run_001")
    parser.add_argument("--tokenizer_ckpt", type=str,
                       default="./experiments/run_001/tokenizer_final.pth",
                       help="Path to tokenizer checkpoint (required for worldmodel stage)")
    parser.add_argument("--data_root", type=str, default=TrainConfig.data_root)
    parser.add_argument("--splits_path", type=str, default=TrainConfig.splits_path)
    parser.add_argument("--img_size", type=int, default=TrainConfig.img_size)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--tok_epochs", type=int, default=TrainConfig.tok_epochs)
    parser.add_argument("--wm_epochs", type=int, default=TrainConfig.wm_epochs)
    parser.add_argument("--lr_tokenizer", type=float, default=TrainConfig.lr_tokenizer)
    parser.add_argument("--lr_worldmodel", type=float, default=TrainConfig.lr_worldmodel)
    parser.add_argument("--vocab_size", type=int, default=TrainConfig.vocab_size)
    parser.add_argument("--embed_dim", type=int, default=TrainConfig.embed_dim)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--save_every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--val_batches", type=int, default=TrainConfig.val_batches)
    parser.add_argument("--resume_tokenizer", type=str, default=None)
    parser.add_argument("--resume_worldmodel", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    exp_dir = Path(cfg.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(
        stride=cfg.stride, seq_len=cfg.seq_len, img_size=cfg.img_size,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, preload=False,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- Build Tokenizer ----
    tokenizer = build_tokenizer(cfg)

    # ---- Stage Selection ----
    if args.stage in ["tokenizer", "both"]:
        # Train tokenizer
        if args.resume_tokenizer:
            ckpt = torch.load(args.resume_tokenizer, map_location=device, weights_only=False)
            tokenizer.load_state_dict(ckpt["tokenizer"])
            print(f"[Resumed Tokenizer] from {args.resume_tokenizer}")

        tokenizer = train_tokenizer(tokenizer, train_loader, cfg, device, exp_dir)

    if args.stage in ["worldmodel", "both"]:
        # Load tokenizer from checkpoint
        tok_path = Path(args.tokenizer_ckpt)
        assert tok_path.exists(), f"Tokenizer checkpoint not found: {tok_path}"
        ckpt = torch.load(tok_path, map_location=device, weights_only=False)
        tokenizer.load_state_dict(ckpt["tokenizer"])
        print(f"[Loaded Tokenizer] from {tok_path}")

        tokenizer = tokenizer.to(device)
        tokenizer.eval()
        for p in tokenizer.parameters():
            p.requires_grad = False

        # Train worldmodel
        worldmodel = train_worldmodel(
            tokenizer, train_loader, val_loader, cfg, device, exp_dir,
            resume_path=args.resume_worldmodel,
        )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Tokenizer: {exp_dir / 'tokenizer_final.pth'}")
    if args.stage in ["worldmodel", "both"]:
        print(f"WorldModel: {exp_dir / 'worldmodel_final.pth'}")


if __name__ == "__main__":
    main()
