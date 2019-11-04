#!/bin/bash

python dataset.py --video_dir /home/hyunho/dataset-sigcomm2020/game-lol/video \
        --image_dir /home/hyunho/dataset-sigcomm2020/game-lol/image \
        --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg/ \
        --video_start_time 0 \
        --video_duration 60 \
        --filter_type uniform \
        --resolution_pairs 240,1080 \
        --load_on_memory
