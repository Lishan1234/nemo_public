#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python train.py --scale 3 --num_patch 50000 --train_data soccer --valid_data soccer --data_type keyframe --data_dir data --num_epoch 1 --num_batch_per_epoch 100
#CUDA_VISIBLE_DEVICES=2 python test.py --scale 4 --train_data starcraft1 --valid_data starcraft1 --data_type 4sec --data_dir data
#CUDA_VISIBLE_DEVICES=3 python train.py --load_on_memory --num_blocks 2 --num_filters 64 --upsample_type transpose --conv_type depthwise_v3
#CUDA_VISIBLE_DEVICES=3 python train.py --load_on_memory --num_blocks 5 --num_filters 64 --upsample_type transpose --conv_type depthwise_v3
