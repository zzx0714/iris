#!/bin/bash
# =============================================================================
# Validation script: run retrieval evaluation on validation set.
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

python validate.py \
    --checkpoint    "$CHECKPOINT" \
    --exp_dir       "$EXP_DIR" \
    --img_size      "$IMG_SIZE" \
    --batch_size    "$BATCH_SIZE" \
    --stride        "$STRIDE" \
    --seq_len       "$SEQ_LEN" \
    --vocab_size    "$VOCAB_SIZE" \
    --embed_dim     "$EMBED_DIM" \
    --device        cuda \
    --num_workers   "$NUM_WORKERS" \
    --output        l1_results.txt \
    "$@"

echo "Validation complete. Results in: ${EXP_DIR}/l1_results.txt"
