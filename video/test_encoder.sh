#!/bin/bash

python encoder.py --dataset_dir /home/hyunho/dataset-sigcomm2020 \
        --content_name game-lol \
        --url https://www.youtube.com/watch?v=BQG92HATfvE \
        --gop 120 \
        --num_threads 4 \
        --start_time 0 \
        --duration 60 \
        --video_fmt webm
