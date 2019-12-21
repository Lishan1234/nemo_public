#!/bin/bash

python encoder.py --video_dir /home/hyunho/dataset-sigcomm2020/game-lol/video \
        --gop 120 \
        --num_threads 4 \
        --start_time 0 \
        --duration 60 \
        --raw_video_fmt webm \
        --video_fmt webm \
        --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg
