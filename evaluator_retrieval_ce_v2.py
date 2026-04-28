"""
IRIS Retrieval Evaluation v2 (group-level, L2 Loss, configurable horizon).

Retrieval pool is defined by background.json groups.
Each group contains ~10 episode-camera pairs (5 episodes × 2 cameras).
For each query, search across ALL frames in the group (not just same episode).

Horizon controls how far ahead the WorldModel predicts:
  h=1: z0_true → predict z1          (N-1 queries per episode)
  h=3: z0..z2_true → predict z3      (N-3 queries per episode)
  h=5: z0..z4_true → predict z5      (N-5 queries per episode)

Teacher forcing: the model always receives ground-truth tokens as context,
matching training.  Prediction is extracted from block (h-1).

Distance metric: L2 (MSE) between predicted logits and one-hot targets.

Usage:
    python evaluator_retrieval_ce_v2.py \
        --tokenizer_checkpoint experiments/run_001/tokenizer_final.pth \
        --worldmodel_checkpoint experiments/run_001/worldmodel_epoch_050.pth \
        --background_json /data3/v-jepa/jepa-wms/evals/l1_retrieval/background.json \
        --horizon 5 \
        --device cuda:1 \
        --output experiments/run_001/iris_retrieval_h5.txt
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
    _load_episode_data, DATASET_ROOT,
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
    # Eval
    horizon        = 5
    top_k          = [1, 5, 10]
    device         = "cuda:1"
    log_all_scores = False


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

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
# Background.json parsing
# ---------------------------------------------------------------------------

CAMERA_KEYS = [
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
]


def _parse_ep_key(key):
    if key.endswith("_2"):
        return int(key[:-2]), 1
    else:
        return int(key[:-2]), 0


def _camera_key_for_idx(cam_idx):
    return CAMERA_KEYS[cam_idx]


# ---------------------------------------------------------------------------
# Load & encode one episode-camera pair
# ---------------------------------------------------------------------------

def _load_and_encode_ep(ep_id, cam_idx, tokenizer, cfg, K, device):
    """Load episode data and encode all frames → tokens (key pool only).

    Returns dict with key_tokens (N, K), actions (N, 7), N — or None.
    WM prediction is deferred to the retrieval phase so we can use the
    correct horizon.
    """
    camera_key = _camera_key_for_idx(cam_idx)
    data = _load_episode_data(ep_id, camera_key, cfg.stride)
    if data is None:
        return None

    frames  = data["frames"]   # (N, H, W, 3) uint8
    actions = data["actions"]  # (N, 7)
    N = len(frames)

    if N < 3:
        return None

    # Resize all frames
    from PIL import Image
    resized = []
    for f in frames:
        img = Image.fromarray(f)
        r = img.resize((cfg.img_size, cfg.img_size), Image.BILINEAR)
        resized.append(np.array(r))
    frames_np = np.stack(resized)  # (N, H, W, 3)

    # Encode ALL frames → tokens (batched to avoid OOM)
    win = cfg.wm_max_blocks
    token_chunks = []
    for i in range(0, N, win):
        chunk = frames_np[i : i + win]
        obs_tensor = (
            torch.from_numpy(chunk).permute(0, 3, 1, 2).float() / 255.0
        )
        obs_tensor = tokenizer.preprocess_input(obs_tensor).to(device)
        enc_out = tokenizer.encode(obs_tensor, should_preprocess=False)
        token_chunks.append(enc_out.tokens.cpu())
    ep_tokens = torch.cat(token_chunks, dim=0)   # (N, K)

    actions_t = torch.from_numpy(actions).float()  # (N, 7)

    return {
        "key_tokens": ep_tokens,   # (N, K)
        "actions": actions_t,      # (N, 7)
        "N": N,
    }


# ---------------------------------------------------------------------------
# Generate queries for one episode with a given horizon
# ---------------------------------------------------------------------------

def _generate_queries(ep_data_dict, worldmodel, cfg, K, device,
                      chunk_size=8):
    """Run WM with autoregressive prediction (no teacher forcing).

    Queries are processed in chunks to avoid OOM.  Each chunk uses its
    own KV cache and runs h autoregressive steps independently.

    Token layout per block: [obs_0, ..., obs_{K-1}, act]
    block_causal mask: within each block all tokens attend to each other,
    so obs tokens can see the action in the same block.

    For horizon h, query starting at frame t:
      Step 0: feed [z_t_true, act[t]]        → predict z_{t+1} (act[t] visible via block_causal)
      Step 1: feed [pred_z_{t+1}, act[t+1]]  → predict z_{t+2} (act[t], act[t+1] visible)
      ...
      Step h-1:                               → predict z_{t+h}  (final logits)
    GT frame index = t + h

    Returns list of {"pred_logits": (K, vocab), "gt_frame_idx": int}.
    """
    h = cfg.horizon
    ep_tokens = ep_data_dict["key_tokens"]   # (N, K)
    actions_t = ep_data_dict["actions"]       # (N, 7)
    N = ep_data_dict["N"]

    if N <= h:
        return []

    B = N - h  # total number of queries

    queries = []

    for chunk_start in range(0, B, chunk_size):
        chunk_end = min(chunk_start + chunk_size, B)
        cb = chunk_end - chunk_start   # chunk batch size

        # Create a fresh KV cache for this chunk
        past_kv = worldmodel.transformer.generate_empty_keys_values(cb)
        past_kv.reset()

        # Initial tokens for this chunk: frames chunk_start..chunk_end-1
        current_tokens = ep_tokens[chunk_start:chunk_end].clone()  # (cb, K)

        for step in range(h):
            obs_input = current_tokens.unsqueeze(1).to(device)    # (cb, 1, K)
            act_idx = chunk_start + step
            act_input = actions_t[act_idx : act_idx + cb].unsqueeze(1).to(device)

            wm_out = worldmodel(obs_input, act_input, past_keys_values=past_kv)
            logits = wm_out.logits_observations   # (cb, K, vocab)

            # Use argmax predicted tokens for next step
            current_tokens = logits.argmax(dim=-1).cpu()   # (cb, K)

        # Final logits: prediction for frame t+h
        final_logits = logits.cpu()    # (cb, K, vocab)

        for i in range(cb):
            queries.append({
                "pred_logits": final_logits[i],   # (K, vocab)
                "gt_frame_idx": chunk_start + i + h,
            })

        # Free GPU memory for this chunk
        del past_kv

    return queries


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_retrieval(
    tokenizer, worldmodel, background_json_path, cfg, K,
    output_path="iris_retrieval_v2_results.txt",
):
    tokenizer.eval()
    worldmodel.eval()
    device = next(tokenizer.parameters()).device
    h = cfg.horizon

    # Load background groups
    with open(background_json_path) as f:
        raw_groups = json.load(f)

    groups = {}
    all_ep_pairs = set()
    for gname, ep_keys in raw_groups.items():
        pairs = []
        for ek in ep_keys:
            ep_id, cam_idx = _parse_ep_key(ek)
            pairs.append((ep_id, cam_idx))
            all_ep_pairs.add((ep_id, cam_idx))
        groups[gname] = pairs

    print(f"Groups: {len(groups)}, unique ep-camera pairs: {len(all_ep_pairs)}")
    print(f"Horizon: {h}")

    # ==================================================================
    # Phase 1: Load & encode all episode-camera pairs (tokens only)
    # ==================================================================
    print("Phase 1: Loading & encoding episodes...")
    ep_data = {}

    pbar = tqdm(all_ep_pairs, desc="Encode")
    for ep_id, cam_idx in pbar:
        result = _load_and_encode_ep(ep_id, cam_idx, tokenizer, cfg, K, device)
        if result is not None:
            ep_data[(ep_id, cam_idx)] = result

    print(f"  Successfully encoded: {len(ep_data)}/{len(all_ep_pairs)}")

    # ==================================================================
    # Phase 2: Group-level retrieval
    # ==================================================================
    print(f"Phase 2: Group-level retrieval (h={h}, metric=L2)...")

    results_lines = []
    hits = {k: 0 for k in cfg.top_k}
    total_queries = 0

    for gname, pairs in tqdm(groups.items(), desc="Groups"):
        group_pairs = [(eid, cid) for eid, cid in pairs
                       if (eid, cid) in ep_data]
        if not group_pairs:
            continue

        # Build combined key pool
        all_keys = []          # (ep_id, cam_idx, local_frame_idx)
        all_key_tokens = []    # each (K,)

        for eid, cid in group_pairs:
            ed = ep_data[(eid, cid)]
            for fi in range(ed["N"]):
                all_keys.append((eid, cid, fi))
                all_key_tokens.append(ed["key_tokens"][fi])

        if not all_key_tokens:
            continue

        key_stack = torch.stack(all_key_tokens)   # (N_group_keys, K)
        N_keys = key_stack.shape[0]

        # Pre-compute key embeddings in codebook space (N_keys, K, embed_dim)
        codebook = tokenizer.embedding.weight.to(device)       # (vocab, embed_dim)
        key_embeds = codebook[key_stack.long().to(device)]      # (N_keys, K, embed_dim)

        g_hits = {k: 0 for k in cfg.top_k}
        g_total = 0

        for eid, cid in group_pairs:
            ed = ep_data[(eid, cid)]
            cam_suffix = "_2" if cid == 1 else "_1"
            ep_label = f"{eid}{cam_suffix}"

            # Generate queries for this episode
            queries = _generate_queries(ed, worldmodel, cfg, K, device)

            for q_item in queries:
                g_total += 1
                pred_logits = q_item["pred_logits"].to(device)   # (K, vocab)
                gt_local = q_item["gt_frame_idx"]

                # Find GT global index
                gt_global = None
                for gi, (ke, kc, kf) in enumerate(all_keys):
                    if ke == eid and kc == cid and kf == gt_local:
                        gt_global = gi
                        break
                if gt_global is None:
                    continue

                # Vectorised L2 in codebook embedding space
                # pred_logits: (K, vocab) → softmax → (K, vocab) → weighted embed (K, embed_dim)
                pred_repr = F.softmax(pred_logits, dim=-1) @ codebook   # (K, embed_dim)
                # key_embeds: (N_keys, K, embed_dim)
                # pred_repr:  (K, embed_dim) → expand to (N_keys, K, embed_dim)
                pred_expanded = pred_repr.unsqueeze(0).expand(N_keys, K, -1)
                l2_per_token = (pred_expanded - key_embeds).pow(2).mean(dim=-1)   # (N_keys, K)
                l2_per_key = l2_per_token.mean(dim=1)                             # (N_keys,)

                # Sort ascending (lower L2 = better)
                sorted_l2, sorted_idx = l2_per_key.sort()

                gt_rank = (sorted_idx == gt_global).nonzero(
                    as_tuple=True)[0].item()

                for k in cfg.top_k:
                    if gt_rank < k:
                        g_hits[k] += 1

                # Log
                if total_queries + g_total <= 500 or cfg.log_all_scores:
                    top_items = []
                    limit = (len(sorted_idx) if cfg.log_all_scores
                             else min(10, len(sorted_idx)))
                    for i in range(limit):
                        gi = sorted_idx[i].item()
                        ke, kc, kf = all_keys[gi]
                        cs = "_2" if kc == 1 else "_1"
                        l2_val = sorted_l2[i].item()
                        top_items.append(f"{ke}{cs}-{kf}({l2_val:.4f})")

                    gt_l2 = l2_per_key[gt_global].item()
                    line = (
                        f"Group_{gname}_{ep_label}_Q{g_total}\t"
                        f"Top10=[{' | '.join(top_items)}]\t"
                        f"GT={ep_label}-{gt_local}({gt_l2:.4f})"
                    )
                    results_lines.append(line)

        for k in cfg.top_k:
            hits[k] += g_hits[k]
        total_queries += g_total

    # ==================================================================
    # Write results
    # ==================================================================
    header = (
        f"IRIS Retrieval Evaluation v2 (group-level, L2 Loss, h={h})\n"
        f"Background: {background_json_path}\n"
        f"Stride: {cfg.stride} | ImgSize: {cfg.img_size} | Horizon: {h}\n"
        + "=" * 80 + "\n"
    )
    summary_parts = [
        f"Hit@{k}: {hits[k] / max(total_queries, 1):.4f}" for k in cfg.top_k
    ]
    summary = (
        f"\n{'=' * 80}\n"
        f"Summary | h={h} | Samples: {total_queries} | "
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
        description="IRIS Retrieval v2 (group-level, L2, configurable horizon)"
    )
    p.add_argument("--tokenizer_checkpoint", type=str, required=True)
    p.add_argument("--worldmodel_checkpoint", type=str, required=True)
    p.add_argument("--background_json", type=str,
                   default="/data3/v-jepa/jepa-wms/evals/l1_retrieval/background.json")
    p.add_argument("--horizon",      type=int, default=5, choices=[1, 3, 5],
                   help="Prediction horizon (1, 3, or 5)")
    p.add_argument("--img_size",     type=int, default=256)
    p.add_argument("--stride",       type=int, default=5)
    p.add_argument("--vocab_size",   type=int, default=1024)
    p.add_argument("--embed_dim",    type=int, default=512)
    p.add_argument("--device",       type=str, default="cuda:1")
    p.add_argument("--output",       type=str, default="")
    p.add_argument("--log_all_scores", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = RetrievalEvalConfig()
    for k in ["img_size", "stride", "vocab_size", "embed_dim",
              "horizon", "log_all_scores"]:
        if hasattr(args, k):
            setattr(cfg, k, getattr(args, k))
    cfg.device = str(device)

    # Default output name includes horizon
    output = args.output
    if not output:
        output = f"experiments/run_001/iris_retrieval_h{cfg.horizon}.txt"

    # Build & load models
    print("Building models...")
    tokenizer, worldmodel, K = build_models(cfg, device)

    print(f"Loading tokenizer: {args.tokenizer_checkpoint}")
    tok_ckpt = torch.load(args.tokenizer_checkpoint,
                          map_location=device, weights_only=False)
    tokenizer.load_state_dict(tok_ckpt["tokenizer"], strict=False)

    print(f"Loading worldmodel: {args.worldmodel_checkpoint}")
    wm_ckpt = torch.load(args.worldmodel_checkpoint,
                         map_location=device, weights_only=False)
    worldmodel.load_state_dict(wm_ckpt["worldmodel"])
    print(f"Loaded. K_obs={K}, vocab_size={cfg.vocab_size}, horizon={cfg.horizon}")

    metrics = evaluate_retrieval(
        tokenizer, worldmodel, args.background_json, cfg, K,
        output_path=output,
    )

    print("\nFinal Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
