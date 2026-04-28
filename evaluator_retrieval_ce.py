"""
IRIS Retrieval Evaluation (per-episode, Cross-Entropy Loss).

Protocol (mirrors evaluator_l1_jepa_wms.py structure):
  Phase 1 — Feature collection:
    For each val episode:
      1. Encode ALL frames with Tokenizer → discrete tokens (key pool).
      2. Run WorldModel once with all frames → predicted logits (queries).
         Block t predicts frame t+1 (causal attention guarantees correctness).
  Phase 2 — Per-episode retrieval:
    For each episode, for each query:
      - Compute CE(pred_logits_t, key_tokens_i) for every key frame i.
      - Sort by CE ascending (lower = better match).
      - Find GT (frame t+1) rank → Hit@k.
  Aggregate hits across all episodes → final Hit@1/5/10.

Usage:
    python evaluator_retrieval_ce.py \
        --checkpoint experiments/run_001/checkpoint_epoch_050.pth \
        --output iris_retrieval_results.txt
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
IRIS_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, IRIS_MODEL_DIR)
sys.path.insert(0, os.path.join(IRIS_MODEL_DIR, ".."))

from src.models import (
    Tokenizer, WorldModel,
    Encoder, Decoder, EncoderDecoderConfig, TransformerConfig,
)
from src.data.droid_dataset import (
    _load_episode_data, _load_meta, DATASET_ROOT, SPLITS_PATH,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class RetrievalEvalConfig:
    """Must match the training configuration."""
    # Tokenizer
    vocab_size       = 1024
    embed_dim        = 512
    ch               = 128
    ch_mult          = [1, 1, 1, 1, 1]
    num_res_blocks   = 2
    attn_resolutions = [16]
    # WorldModel
    wm_num_layers = 10
    wm_num_heads  = 4
    wm_embed_dim  = 256
    wm_max_blocks = 20
    act_dim       = 7
    # Data
    img_size    = 256
    stride      = 5
    camera_key  = "observation.images.exterior_image_1_left"
    # Eval
    top_k          = [1, 5, 10]
    device         = "cuda"
    max_episodes   = None
    log_all_scores = False


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_models(cfg: RetrievalEvalConfig, device: torch.device):
    """Build Tokenizer + WorldModel, return (tokenizer, worldmodel, K)."""
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
        tokens_per_block=K + 1,
        max_blocks=cfg.wm_max_blocks,
        attention="block_causal",
        num_layers=cfg.wm_num_layers,
        num_heads=cfg.wm_num_heads,
        embed_dim=cfg.wm_embed_dim,
        embed_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    worldmodel = WorldModel(
        obs_vocab_size=cfg.vocab_size,
        config=tcfg,
        act_dim=cfg.act_dim,
    ).to(device)

    return tokenizer, worldmodel, K


# ---------------------------------------------------------------------------
# Val episode IDs
# ---------------------------------------------------------------------------

def get_val_episode_ids(splits_path: str = SPLITS_PATH):
    """Load val episode IDs from splits JSON."""
    with open(splits_path) as f:
        splits = json.load(f)

    val_ids = sorted(int(x) for x in splits["val"])
    return val_ids


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_retrieval_ce(
    tokenizer: Tokenizer,
    worldmodel: WorldModel,
    val_episode_ids,
    cfg: RetrievalEvalConfig,
    K: int,
    output_path: str = "iris_retrieval_results.txt",
):
    tokenizer.eval()
    worldmodel.eval()
    device = next(tokenizer.parameters()).device

    max_frames = cfg.wm_max_blocks  # cap per episode to model context length

    results_lines = []
    hits = {k: 0 for k in cfg.top_k}
    total_queries = 0
    episodes_done = 0

    pbar = tqdm(val_episode_ids, desc="Retrieval")

    for ep_id in pbar:
        if cfg.max_episodes and episodes_done >= cfg.max_episodes:
            break

        # ------------------------------------------------------------------
        # Load full episode
        # ------------------------------------------------------------------
        data = _load_episode_data(ep_id, cfg.camera_key, cfg.stride)
        if data is None:
            continue

        frames  = data["frames"]   # (N, H, W, 3) uint8
        actions = data["actions"]  # (N, 7) float32
        N = len(frames)

        if N < 3:
            continue

        # Cap to max_frames so the WorldModel can process in one pass
        if N > max_frames:
            N = max_frames
            frames  = frames[:N]
            actions = actions[:N]

        # Resize to model resolution
        from PIL import Image
        resized = []
        for f in frames:
            img = Image.fromarray(f)
            r = img.resize((cfg.img_size, cfg.img_size), Image.BILINEAR)
            resized.append(np.array(r))
        frames_np = np.stack(resized)  # (N, H, W, 3)

        # ==================================================================
        # Phase 1: Encode + Predict
        # ==================================================================

        # Encode all frames → discrete tokens  (key pool)
        obs_tensor = (
            torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        )                                                          # (N, 3, H, W) [0,1]
        obs_tensor = tokenizer.preprocess_input(obs_tensor).to(device)  # [-1,1]
        enc_out = tokenizer.encode(obs_tensor, should_preprocess=False)
        ep_tokens = enc_out.tokens.cpu()                           # (N, K)  int64

        actions_t = torch.from_numpy(actions).float()              # (N, 7)

        # WorldModel forward — one pass for the whole episode
        #   Block t predicts frame t+1 (causal attention guarantees this).
        ctx_obs = ep_tokens.unsqueeze(0).to(device)                # (1, N, K)
        ctx_act = actions_t.unsqueeze(0).to(device)                # (1, N, 7)
        wm_out = worldmodel(ctx_obs, ctx_act)
        all_logits = wm_out.logits_observations.cpu()              # (1, N*K, vocab)

        # ==================================================================
        # Phase 2: Per-episode retrieval
        # ==================================================================

        key_tokens = ep_tokens  # (N, K)

        ep_hits = {k: 0 for k in cfg.top_k}
        ep_queries = 0

        for t in range(N - 1):
            # Block t predicts frame t+1
            pred_logits = all_logits[0, t * K : (t + 1) * K]      # (K, vocab)
            gt_frame_idx = t + 1

            # --- Vectorised CE with all key frames ---
            # pred_logits: (K, vocab)  →  (N, K, vocab)
            expanded = pred_logits.unsqueeze(0).expand(N, K, -1)
            flat_logits  = expanded.reshape(N * K, cfg.vocab_size)
            flat_targets = key_tokens.long().reshape(N * K)
            ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            ce_per_frame = ce_per_token.reshape(N, K).mean(dim=1)  # (N,)

            # Sort ascending: lower CE = better match
            sorted_ces, sorted_idx = ce_per_frame.sort()

            # GT rank
            gt_rank = (sorted_idx == gt_frame_idx).nonzero(as_tuple=True)[0].item()

            for k in cfg.top_k:
                if gt_rank < k:
                    ep_hits[k] += 1
            ep_queries += 1

            # Detailed log (first 500 queries globally, or all if requested)
            if total_queries + ep_queries <= 500 or cfg.log_all_scores:
                top_items = []
                limit = len(sorted_idx) if cfg.log_all_scores else min(10, len(sorted_idx))
                for i in range(limit):
                    idx_val = sorted_idx[i].item()
                    ce_val  = sorted_ces[i].item()
                    top_items.append(f"{idx_val}({ce_val:.4f})")
                gt_ce = ce_per_frame[gt_frame_idx].item()
                line = (
                    f"Ep_{ep_id:06d}_Q{ep_queries}\t"
                    f"Top10=[{' | '.join(top_items)}]\t"
                    f"GT={gt_frame_idx}({gt_ce:.4f})"
                )
                results_lines.append(line)

        # Accumulate across episodes
        for k in cfg.top_k:
            hits[k] += ep_hits[k]
        total_queries += ep_queries
        episodes_done += 1

        pbar.set_postfix({
            "ep": episodes_done,
            "queries": total_queries,
            "H@1": f"{hits[1]/max(total_queries,1):.3f}",
        })

    # ======================================================================
    # Write results
    # ======================================================================
    header = (
        "IRIS Retrieval Evaluation (per-episode, CE Loss)\n"
        f"Stride: {cfg.stride} | ImgSize: {cfg.img_size} | MaxFrames: {max_frames}\n"
        + "=" * 80 + "\n"
    )
    summary_parts = [
        f"Hit@{k}: {hits[k] / max(total_queries, 1):.4f}" for k in cfg.top_k
    ]
    summary = (
        f"\n{'=' * 80}\n"
        f"Summary | Episodes: {episodes_done} | Samples: {total_queries} | "
        + " | ".join(summary_parts) + "\n"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(header)
        for line in results_lines:
            f.write(line + "\n")
        f.write(summary)

    metrics = {f"Hit@{k}": hits[k] / max(total_queries, 1) for k in cfg.top_k}
    print(summary)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="IRIS Retrieval Evaluation (per-episode, CE Loss)"
    )
    p.add_argument("--tokenizer_checkpoint", type=str, required=True,
                   help="Path to tokenizer checkpoint (e.g. tokenizer_final.pth)")
    p.add_argument("--worldmodel_checkpoint", type=str, required=True,
                   help="Path to worldmodel checkpoint (e.g. worldmodel_epoch_050.pth)")
    p.add_argument("--img_size",    type=int, default=256)
    p.add_argument("--stride",      type=int, default=5)
    p.add_argument("--vocab_size",  type=int, default=1024)
    p.add_argument("--embed_dim",   type=int, default=512)
    p.add_argument("--device",      type=str, default="cuda:1")
    p.add_argument("--output",      type=str, default="iris_retrieval_results.txt")
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--log_all_scores", action="store_true")
    p.add_argument("--splits_path", type=str, default=SPLITS_PATH)
    p.add_argument("--camera_key",  type=str,
                   default="observation.images.exterior_image_1_left")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = RetrievalEvalConfig()
    for k in [
        "img_size", "stride", "vocab_size", "embed_dim",
        "camera_key", "max_episodes", "log_all_scores",
    ]:
        if hasattr(args, k):
            setattr(cfg, k, getattr(args, k))
    cfg.device = str(device)

    # Build & load models
    print("Building models...")
    tokenizer, worldmodel, K = build_models(cfg, device)

    print(f"Loading tokenizer: {args.tokenizer_checkpoint}")
    tok_ckpt = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    tokenizer.load_state_dict(tok_ckpt["tokenizer"], strict=False)

    print(f"Loading worldmodel: {args.worldmodel_checkpoint}")
    wm_ckpt = torch.load(args.worldmodel_checkpoint, map_location=device, weights_only=False)
    worldmodel.load_state_dict(wm_ckpt["worldmodel"])
    print(f"Loaded. K_obs={K}, vocab_size={cfg.vocab_size}")

    # Val episodes
    val_ids = get_val_episode_ids(args.splits_path)
    print(f"Val episodes: {len(val_ids)}")

    # Run
    metrics = evaluate_retrieval_ce(
        tokenizer, worldmodel, val_ids, cfg, K,
        output_path=args.output,
    )

    print("\nFinal Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
