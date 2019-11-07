#!/bin/bash
python test_s.py --dataset_dir /home/hyunho/dataset-sigcomm2020/game-lol \
                --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
                --start_time 0 \
                --duration 60 \
                --input_resolution 240 \
                --target_resolution 1080 \
                --num_filters 32 \
                --num_blocks 4 \
                --model_type edsr_s \
                --load_on_memory 
