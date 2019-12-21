#!/bin/bash
python ../visualize_graph_ed_s.py --dataset_dir /ssd1/dataset-sigcomm2020/game-lol \
                --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
                --start_time 0 \
                --duration 60 \
                --input_resolution 240 \
                --target_resolution 1080 \
                --enc_num_filters 64 \
                --enc_num_blocks 8 \
                --dec_num_filters 32 \
                --dec_num_blocks 2 
