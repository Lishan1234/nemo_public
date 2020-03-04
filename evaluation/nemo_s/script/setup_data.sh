#!/bin/bash

python $NEMO_CODE_ROOT/evaluation/nemo_s/setup_data.py \
    --dataset_dir $NEMO_DATA_ROOT/unboxing \
    --lr_video_name 240p_s0_d300_encoded.webm \
    --upsample_type deconv \
    --num_filters 32 \
    --num_blocks 8 \
    --device_id 7b7f59d1 \
    --threshold 0.2
