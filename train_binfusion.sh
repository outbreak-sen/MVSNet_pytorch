#!/bin/bash
# Training script for Depth Bin Fusion Model

# Configuration
TRAINPATH="/data1/local_userdata/houbosen/dtu_training_raw"
TESTPATH="/data1/local_userdata/houbosen/dtu_training_raw"
TRAINLIST="lists/dtu/train.txt"
TESTLIST="lists/dtu/val.txt"
MVSNET_CKPT="checkpoints/model_000014.ckpt"
LOGDIR="checkpoints/fusion"

# Training parameters
EPOCHS=20
BATCH_SIZE=4
LR=0.0001
LR_EPOCHS="10,15,18:2"
NUM_DEPTH=192
INTERVAL_SCALE=1.06
NUM_BINS=64
HIDDEN_DIM=64

# Run training
python train_fusion.py \
    --trainpath ${TRAINPATH} \
    --testpath ${TESTPATH} \
    --trainlist ${TRAINLIST} \
    --testlist ${TESTLIST} \
    --mvsnet_ckpt ${MVSNET_CKPT} \
    --logdir ${LOGDIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --lrepochs ${LR_EPOCHS} \
    --numdepth ${NUM_DEPTH} \
    --interval_scale ${INTERVAL_SCALE} \
    --num_bins ${NUM_BINS} \
    --hidden_dim ${HIDDEN_DIM} \
    --summary_freq 20 \
    --save_freq 1 \
    --seed 1
