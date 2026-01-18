#!/bin/bash

# ResidualFusionNet 快速训练命令

python train_residual_fusion.py \
    --trainpath /data1/local_userdata/houbosen/dtu_training_raw \
    --testpath /data1/local_userdata/houbosen/dtu_training_raw \
    --trainlist lists/dtu/train.txt \
    --testlist lists/dtu/val.txt \
    --mvsnet_ckpt checkpoints/model_000014.ckpt \
    --epochs 20 \
    --batch_size 4 \
    --lr 0.0001 \
    --hidden_dim 48 \
    --lambda_cons_base 0.1 \
    --lambda_uncert 1.0 \
    --cons_weight_type exp \
    --logdir ./checkpoints/residual_fusion \
    --summary_freq 20 \
    --save_freq 1
