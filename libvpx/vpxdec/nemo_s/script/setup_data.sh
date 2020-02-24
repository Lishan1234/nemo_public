#!/bin/bash

python $MOBINAS_CODE_ROOT/libvpx/vpxdec/nemo_s/setup_data.py --dataset_dir $MOBINAS_DATA_ROOT/how_to \
    --lr_video_name 240p_s0_d300_encoded.webm \
    --hr_video_name 960p_240p_s0_d300.webm \
    --upsample_type deconv \
    --num_filters 32 \
    --num_blocks 8 \
    --device_id a152b92a \
    --aps_class nemo \
    --threshold 0.5 \
    --limit 240
