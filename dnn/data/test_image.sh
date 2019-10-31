#!/bin/bash

python image.py --video_dir /home/hyunho/dataset-sigcomm2020/game-lol/video \
        --image_dir /home/hyunho/dataset-sigcomm2020/game-lol/image \
        --lr_video_name 240p_s0_d60_encoded \
        --hr_video_name 1080p_s0_d60 \
        --video_fmt webm \
        --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
        --filter_frames uniform
