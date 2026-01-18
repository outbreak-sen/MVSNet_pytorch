#!/bin/bash
# Training script for Residual Fusion Model

TRAINPATH="/data1/local_userdata/houbosen/dtu_training_raw"
TESTPATH="/data1/local_userdata/houbosen/dtu_training_raw"
TRAINLIST="lists/dtu/train.txt"
TESTLIST="lists/dtu/val.txt"
MVSNET_CKPT="checkpoints/model_000014.ckpt"
LOGDIR="checkpoints/residual_fusion"

python train_residual_fusion.py \
    --trainpath ${TRAINPATH} \
    --testpath ${TESTPATH} \
    --trainlist ${TRAINLIST} \
    --testlist ${TESTLIST} \
    --mvsnet_ckpt ${MVSNET_CKPT} \
    --logdir ${LOGDIR} \
    --epochs 20 \
    --batch_size 4 \
    --lr 0.0001 \
    --lrepochs "10,15,18:2" \
    --numdepth 192 \
    --interval_scale 1.06 \
    --hidden_dim 48 \
    --summary_freq 20 \
    --save_freq 1 \
    --seed 1
