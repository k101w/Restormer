#!/usr/bin/env bash

CONFIG=$1
export NCCL_DEBUG=INFO
torchrun --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch 