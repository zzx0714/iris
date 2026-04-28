#!/bin/bash
# =============================================================================
# Inference script: load weights and run retrieval on validation set.
# =============================================================================

set -e

EXP_DIR="./experiments/run_001"
CHECKPOINT="${EXP_DIR}/checkpoint_epoch_010.pth"

# Config (must match training)
IMG_SIZE=256
STRIDE=5
SEQ_LEN=5
BATCH_SIZE=8
VOCAB_SIZE=1024
EMBED_DIM=256
NUM_WORKERS=4
MAX_EPISODES=500   # set to null to evaluate all episodes

python inference.py \
    --checkpoint    "$CHECKPOINT" \
    --exp_dir       "$EXP_DIR" \
    --img_size      "$IMG_SIZE" \
    --stride        "$STRIDE" \
    --seq_len       "$SEQ_LEN" \
    --batch_size    "$BATCH_SIZE" \
    --vocab_size    "$VOCAB_SIZE" \
    --embed_dim     "$EMBED_DIM" \
    --device        cuda \
    --num_workers   "$NUM_WORKERS" \
    --max_episodes  "$MAX_EPISODES" \
    --output        inference_results.txt \
    "$@"

echo "Inference complete. Results in: ${EXP_DIR}/inference_results.txt"
