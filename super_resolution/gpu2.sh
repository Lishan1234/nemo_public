#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python train.py --scale 4 --num_patch 50000 --train_data starcraft1 --valid_data starcraft1 --data_type keyframe --data_dir /ssd1 
CUDA_VISIBLE_DEVICES=2 python test.py --scale 4 --train_data starcraft1 --valid_data starcraft1 --data_type 4sec --data_dir data
