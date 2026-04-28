#!/bin/bash
# =============================================================================
# Training script for IRIS Tokenizer + WorldModel on DROID dataset.
# Separate training: Stage 1 (Tokenizer) -> Stage 2 (WorldModel)
# =============================================================================

set -e

DATA_ROOT="/data3/v-jepa/jepa-wms/dataset/droid_lerobot_dataset"
SPLITS_PATH="/data3/v-jepa/jepa-wms/1w2splits.json"
EXP_DIR="./experiments/run_001"

# Image resolution
IMG_SIZE=256

# Tokenizer hyperparameters
VOCAB_SIZE=1024
EMBED_DIM=256
CH=128
CH_MULT="1,2,4,4"
NUM_RES_BLOCKS=2

# WorldModel hyperparameters
WM_NUM_LAYERS=6
WM_NUM_HEADS=8
WM_EMBED_DIM=256

# Training hyperparameters
BATCH_SIZE=8
TOK_EPOCHS=50
WM_EPOCHS=50
LR_TOKENIZER=3e-4
LR_WORLDMODEL=1e-4
NUM_WORKERS=4
SAVE_EVERY=5

# -----------------------------------------------------------------------
# Use GPU
# -----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0

# -----------------------------------------------------------------------
# Stage 1: Train Tokenizer
# -----------------------------------------------------------------------
echo "=========================================="
echo "STAGE 1: TRAINING TOKENIZER"
echo "=========================================="

python train.py \
    --stage tokenizer \
    --exp_dir          "$EXP_DIR" \
    --data_root         "$DATA_ROOT" \
    --splits_path       "$SPLITS_PATH" \
    --img_size          "$IMG_SIZE" \
    --batch_size        "$BATCH_SIZE" \
    --tok_epochs        "$TOK_EPOCHS" \
    --lr_tokenizer      "$LR_TOKENIZER" \
    --vocab_size        "$VOCAB_SIZE" \
    --embed_dim         "$EMBED_DIM" \
    --device            cuda \
    --num_workers       "$NUM_WORKERS" \
    --save_every        "$SAVE_EVERY"

echo "Tokenizer training complete!"
echo "Checkpoint saved to: $EXP_DIR/tokenizer_final.pth"

# -----------------------------------------------------------------------
# Stage 2: Train WorldModel (with frozen tokenizer)
# -----------------------------------------------------------------------
echo ""
echo "=========================================="
echo "STAGE 2: TRAINING WORLDMODEL"
echo "=========================================="

python train.py \
    --stage worldmodel \
    --exp_dir          "$EXP_DIR" \
    --tokenizer_ckpt   "$EXP_DIR/tokenizer_final.pth" \
    --data_root         "$DATA_ROOT" \
    --splits_path       "$SPLITS_PATH" \
    --img_size          "$IMG_SIZE" \
    --batch_size        "$BATCH_SIZE" \
    --wm_epochs         "$WM_EPOCHS" \
    --lr_worldmodel     "$LR_WORLDMODEL" \
    --vocab_size        "$VOCAB_SIZE" \
    --embed_dim         "$EMBED_DIM" \
    --device            cuda \
    --num_workers       "$NUM_WORKERS" \
    --save_every        "$SAVE_EVERY"

echo "=========================================="
echo "TRAINING COMPLETE!"
echo "=========================================="
echo "Tokenizer: $EXP_DIR/tokenizer_final.pth"
echo "WorldModel: $EXP_DIR/worldmodel_final.pth"
