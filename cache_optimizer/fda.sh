#!/bin/sh
python frame_dependency_analyzer.py --vpxdec_path /home/hyunho/MobiNAS/third_party/libvpx/vpxdec \
                                --content_dir /ssd1/data-sigcomm2020/game-lol \
                                --input_video_name 240p_s0_d60_encoded.webm \
                                --num_threads 1 \
                                --gop 120 
