#!/bin/bash

python image.py --dataset_dir /home/hyunho/dataset-sigcomm2020 \
        --content_name game-lol \
        --start_time 0 \
        --duration 60 \
        --video_fmt webm \
        --resolution_pairs 240,1080 \
        --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
        --filter_frames uniform
