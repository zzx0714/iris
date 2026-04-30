"""
Retrieval evaluation module.
Provides retrieval evaluation logic for the Tokenizer + WorldModel pipeline.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class RetrievalConfig:
    """Default config for retrieval evaluation (must match TrainConfig)."""
    img_size     = 256
    stride       = 5
    seq_len      = 5
    batch_size   = 1  # One episode per batch for retrieval
    num_workers  = 4
    vocab_size   = 1024
    embed_dim    = 512
    ch           = 128
    ch_mult      = [1, 1, 1, 1, 1]
    num_res_blocks = 2
    attn_resolutions = [16]
    wm_num_layers = 10
    wm_num_heads  = 8
    wm_embed_dim  = 512
    wm_max_blocks = 20
    act_dim       = 7
    device       = "cuda"


def _to_device(batch, device):
    """Move batch tensors to device."""
    def _t(x):
        return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
    from src.batch import Batch
    return Batch(
        observations  = _t(batch.observations),
        actions       = _t(batch.actions),
        states        = _t(batch.states),
        mask_padding  = _t(batch.mask_padding),
        episode_idx   = batch.episode_idx,
        frame_idx     = _t(batch.frame_idx) if batch.frame_idx is not None else None,
    )


@torch.no_grad()
def run_retrieval(
    tokenizer,
    worldmodel,
    val_loader: DataLoader,
    device: torch.device,
    cfg: RetrievalConfig,
    K: int,
    output_path: str,
    top_k: list = [1, 5, 10],
    max_episodes: int = None,
    verbose: bool = True,
):
    """
    Per-episode retrieval evaluation.
    
    For each query frame t, predict frame t+stride using WorldModel,
    then compare predicted tokens against ALL keys in the episode.
    
    Query indexing convention:
        - Q1 corresponds to query at frame 0, goal frame = frame stride
        - e.g., with stride=5: Q1->goal=5, Q2->goal=6, etc.
    
    Args:
        tokenizer: Tokenizer model
        worldmodel: WorldModel
        val_loader: DataLoader for validation episodes
        device: torch device
        cfg: RetrievalConfig
        K: Number of tokens per frame (latent_h * latent_w)
        output_path: Path to save results
        top_k: List of k values for Hit@k
        max_episodes: Maximum number of episodes to evaluate
        verbose: Whether to show progress bar
    
    Returns:
        Dictionary of Hit@k metrics
    """
    tokenizer.eval()
    worldmodel.eval()

    results_lines = []
    hits = {k: 0 for k in top_k}
    total_queries = 0
    episodes_done = 0

    if verbose:
        pbar = tqdm(val_loader, desc="Retrieval")

    for batch in (pbar if verbose else val_loader):
        if max_episodes and episodes_done >= max_episodes:
            break
        batch = _to_device(batch, device)
        B, T, C, H, W = batch.observations.shape
        obs = batch.observations.float() / 255.0   # (B, T, 3, H, W)

        for b in range(B):
            if max_episodes and episodes_done >= max_episodes:
                break

            ep_id = batch.episode_idx[b].item()

            # ---- Encode all frames → discrete tokens ----
            obs_flat = obs[b].reshape(T, C, H, W)            # (T, 3, H, W)
            enc_out = tokenizer.encode(
                tokenizer.preprocess_input(obs_flat),
                should_preprocess=False,
            )
            ep_tokens = enc_out.tokens   # (T, K)

            # Key pool = all real tokens in the episode
            key_tokens = ep_tokens   # (T, K)

            actions_ep = batch.actions[b]   # (T, act_dim)
            mask_ep    = batch.mask_padding[b]  # (T,)

            # ---- For each query t, predict frame t+stride ----
            # Query Q{t+1} uses context frames 0..t, predicts frame t+stride
            # Valid queries require at least `stride` frames of context (t >= stride-1)
            # So earliest valid query is Q_stride (t = stride - 1)
            for t in range(cfg.stride - 1, T - cfg.stride):
                if not mask_ep[t] or not mask_ep[t + cfg.stride]:
                    continue

                # Context: frames 0..t (inclusive)
                ctx_obs = ep_tokens[: t + 1]         # (t+1, K)
                ctx_act = actions_ep[: t + 1]       # (t+1, act_dim)

                # WorldModel forward
                ctx_obs_b = ctx_obs.unsqueeze(0)    # (1, t+1, K)
                ctx_act_b = ctx_act.unsqueeze(0)    # (1, t+1, act_dim)

                wm_out = worldmodel(ctx_obs_b, ctx_act_b)
                logits_obs = wm_out.logits_observations   # (1, (t+1)*K, vocab)

                # Block t predicts frame t+1, so for predicting frame t+stride,
                # we need block at position (t+stride-1)
                pred_block_start = (t + cfg.stride - 1) * K
                pred_logits = logits_obs[0, pred_block_start : pred_block_start + K]   # (K, vocab_size)

                # ---- Retrieval: CE loss with all key tokens ----
                # For each key frame, compute average CE across all K positions
                ce_losses = []
                for kt_idx, kt in enumerate(key_tokens):   # (K,) each
                    ce_k = F.cross_entropy(
                        pred_logits,                    # (K, vocab)
                        kt.long().expand(K),           # (K,)
                        reduction="none",
                    ).mean().item()
                    ce_losses.append(ce_k)

                ce_losses_t = torch.tensor(ce_losses)
                sorted_ces, sorted_idx = ce_losses_t.sort()   # ascending: smallest = best

                # Ground truth: frame at index t + stride
                gt_idx = t + cfg.stride
                gt_rank = (sorted_idx == gt_idx).nonzero(as_tuple=True)[0].item()

                for kk in top_k:
                    if gt_rank < kk:
                        hits[kk] += 1
                total_queries += 1

                # Format Top10 list
                top10_items = []
                for i in range(min(10, len(sorted_idx))):
                    frame_idx_val = sorted_idx[i].item()
                    ce_val = sorted_ces[i].item()
                    top10_items.append(f"{frame_idx_val}({ce_val:.4f})")
                top10_str = " | ".join(top10_items)

                gt_ce = ce_losses[gt_idx]
                ep_str = f"{ep_id:06d}"
                query_num = t + 1  # Q1, Q2, ... (1-indexed)
                line = (
                    f"Ep_{ep_str}_Q{query_num}\t"
                    f"Top10=[{top10_str}]\t"
                    f"GT={gt_idx}({gt_ce:.4f})"
                )
                results_lines.append(line)

            episodes_done += 1
            if verbose and pbar:
                pbar.set_postfix({"ep": episodes_done, "queries": total_queries})

    # ---- Write results ----
    top_k_strs = [f"Hit@{k}: {hits[k]/max(total_queries,1):.4f}" for k in top_k]
    header = (
        "JEPA-WMS L1 Retrieval Evaluation\n"
        "Metric: Cross-Entropy Loss\n"
        "================================================================================\n"
    )
    summary = (
        f"\n================================================================================\n"
        f"Summary | Samples: {total_queries} | " + " | ".join(top_k_strs) + "\n"
    )

    with open(output_path, "w") as f:
        f.write(header)
        for line in results_lines:
            f.write(line + "\n")
        f.write(summary)

    if verbose:
        print(f"\n[Results: {output_path}]")
        print(f"Summary | Samples: {total_queries} | " + " | ".join(top_k_strs))

    return {f"Hit@{k}": hits[k] / max(total_queries, 1) for k in top_k}
