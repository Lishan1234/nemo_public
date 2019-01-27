#!/bin/bash
#CUDA_VISIBLE_DEVICES=1 python train.py --scale 3 --num_patch 50000 --train_data soccer --valid_data soccer 
CUDA_VISIBLE_DEVICES=1 python train.py --scale 3 --num_patch 50000 --train_data starcraft1 --valid_data starcraft1
