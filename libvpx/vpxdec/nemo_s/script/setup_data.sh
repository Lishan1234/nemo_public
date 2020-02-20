#!/bin/bash

python $MOBINAS_CODE_ROOT/libvpx/vpxdec/nemo_s/setup_data.py --dataset_dir $MOBINAS_DATA_ROOT/education \
    --lr_video_name 480p_s0_d300_encoded.webm \
    --hr_video_name 1080p_s0_d300.webm \
    --upsample_type deconv \
    --num_filters 26 \
    --num_blocks 8 \
    --device_id a152b92a \
    --limit 30 
