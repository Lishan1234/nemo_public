#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --scale 2 --num_patch 50000 --train_data starcraft1 --valid_data starcraft1 --data_type keyframe --data_dir /ssd1 
CUDA_VISIBLE_DEVICES=0 python test.py --scale 2 --train_data starcraft1 --valid_data starcraft1 --data_type 4sec --data_dir data
#CUDA_VISIBLE_DEVICES=0 python test.py --scale 2 --num_patch 50000 --train_data starcraft1 --valid_data starcraft1 --data_type 4sec
