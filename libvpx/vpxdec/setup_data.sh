#!/bin/bash

python setup_data.py --video_dir $HOME/data-sigcomm2020/game-lol/video/ \
    --checkpoint_dir $HOME/data-sigcomm2020/game-lol/checkpoint/240p_s0_d60_encoded.webm.uniform_1.00/ \
    --lr_video_name 240p_s0_d60_encoded.webm \
    --hr_video_name 1080p_s0_d60.webm \
    --num_filters 32 \
    --num_blocks 4 \
    --scale 4 \
    --nhwc 1 240 426 3 \
    --device_id a152b92a \
    --threads 1 \
    --limit 30 
