#!/bin/bash

python setup_build.py --video_dir $HOME/data-sigcomm2020/game-lol/video/ \
    --lr_video_name 270p_512k_60sec_125st.webm \
    --hr_video_name 1080p_lossless_60sec_125st.webm \
    --abi arm64-v8a \
    --device_id a152b92a 
