#!/bin/sh
python anchor_point_selector_baseline.py --vpxdec_path /home/hyunho/MobiNAS/third_party/libvpx/vpxdec \
                                --content_dir /ssd1/data-sigcomm2020/game-lol \
                                --input_video_name 240p_s0_d60_encoded.webm \
                                --compare_video_name 960p_s0_d60.webm \
                                --num_decoders 16 \
                                --gop 120 \
                                --num_blocks 8 \
                                --num_filters 64 \
                                --checkpoint_dir /ssd1/data-sigcomm2020/game-lol/checkpoint/240p_s0_d60_encoded.webm.uniform_1.00 \
                                --quality_diff 0.2
