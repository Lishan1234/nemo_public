#!/bin/bash

python $MOBINAS_CODE_ROOT/libvpx/vpxdec/nemo_s/setup_data.py --dataset_dir $MOBINAS_DATA_ROOT/product_review \
    --lr_video_name 360p_s0_d300_encoded.webm \
    --hr_video_name 1080p_s0_d300.mp4 \
    --upsample_type deconv \
    --num_filters 48 \
    --num_blocks 8 \
    --device_id a152b92a \
    --limit 30 
