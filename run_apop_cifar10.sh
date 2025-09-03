#!/bin/bash
cd system

python main.py \
    -data Cifar10 \
    -m ResNet18 \
    -algo APOP \
    -gr 200 \
    -nc 5 \
    -cil True \
    -til True \
    -pfcl True \
    -client_seq "0:0,1|2,3|4,5|6,7|8,9;1:2,3|4,5|6,7|8,9|0,1;2:4,5|6,7|8,9|0,1|2,3;3:6,7|8,9|0,1|2,3|4,5;4:8,9|0,1|2,3|4,5|6,7" \
    -cilrpc 40 \
    -subspace_dim 25 \
    -adaptation_threshold 0.4 \
    -fusion_threshold 0.45 \
    -max_transfer_gain 1.8 \
    -lr 0.01 \
    -lbs 32 \
    -ls 3 \
    -eg 5 \
    -wandb True \
    -wandb_project "apop1" \
    -go "apop_cifar10_resnet18_5clients_comprehensive_demo"
