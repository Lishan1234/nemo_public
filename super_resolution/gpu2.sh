#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python train.py --scale 4 --num_patch 50000 --train_data soccer --valid_data soccer --data_type keyframe --data_dir /ssd1 --num_epoch 1
CUDA_VISIBLE_DEVICES=2 python test.py --scale 4 --train_data soccer --valid_data soccer --data_type 4sec --data_dir data
