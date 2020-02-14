#!/bin/sh
python logger.py --content_dir /ssd1/data-sigcomm2020/game-lol \
                --input_video_name 240p_s0_d60_encoded.webm \
                --compare_video_name 960p_s0_d60.webm \
                --gop 120 \
                --num_blocks 8 \
                --num_filters 64 \
                --quality_diff 0.2 
