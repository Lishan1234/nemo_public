#!/bin/bash

#CUDA_VISIBLE_DEVICES=3 python train.py --load_on_memory --num_blocks 2 --num_filters 64 --upsample_type transpose --conv_type depthwise_v3
CUDA_VISIBLE_DEVICES=3 python train.py --load_on_memory --num_blocks 5 --num_filters 64 --upsample_type transpose --conv_type depthwise_v3
