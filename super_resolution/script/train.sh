#!/bin/sh

CUDA_VISIBLE_DEVICES=gpu_index python train.py --num_blocks num_blocks --num_filters num_filters --train_data train_data --train_datatype train_datatype --scale scale
