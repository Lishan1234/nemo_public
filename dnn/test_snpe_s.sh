#!/bin/bash
python snpe_s.py --dataset_dir /home/hyunho/dataset-sigcomm2020/game-lol \
                --snpe_dir /home/hyunho/snpe-1.32.0.555 \
                --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
                --start_time 0 \
                --duration 60 \
                --input_resolution 240 \
                --target_resolution 1080 \
                --num_filters 32 \
                --num_blocks 4 \
                --model_type edsr_s \
                --hwc 240 426 3
