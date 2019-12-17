#!/bin/bash

python setup_data.py --video_dir $HOME/data-sigcomm2020/game-lol/video/ \
    --checkpoint_dir $HOME/data-sigcomm2020/game-lol/checkpoint/240p_s0_d60_encoded.webm.uniform_1.00/ \
    --lr_video_name 270p_512k_60sec_125st.webm \
    --hr_video_name 1080p_lossless_60sec_125st.webm \
    --num_filters 32 \
    --num_blocks 4 \
    --scale 4 \
    --nhwc 1 270 480 3 \
    --device_id a152b92a \
    --threads 1 \
    --limit 30 
