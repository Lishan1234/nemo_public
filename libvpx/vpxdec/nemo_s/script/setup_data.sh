#!/bin/bash

python $MOBINAS_CODE_ROOT/libvpx/vpxdec/nemo_s/setup_data.py --dataset_dir $MOBINAS_DATA_ROOT/game-lol \
    --lr_video_name 240p_s0_d60_encoded.webm \
    --hr_video_name 960p_s0_d60.webm \
    --upsample_type deconv \
    --num_filters 48 \
    --num_blocks 8 \
    --device_id a152b92a \
    --limit 30 
