#!/bin/bash
python multi_benchmark.py --num_blocks 8 --num_filters 64 --model_type edsr_v2 --num_reduced_filters 3 --upsample_type transpose --train_data news --lr 240
python multi_benchmark.py --num_blocks 8 --num_filters 64 --model_type edsr_v2 --num_reduced_filters 3 --upsample_type resize_bilinear --train_data news --lr 240
