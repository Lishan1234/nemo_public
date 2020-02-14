#!/bin/bash
python $DNNDIR/visualize_graph_ed_s_v2.py --dataset_dir $DATADIR/game-lol \
                --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
                --start_time 0 \
                --duration 60 \
                --input_resolution 240 \
                --target_resolution 1080 \
                --enc_num_filters 64 \
                --enc_num_blocks 8 \
                --dec_num_filters 16 \
                --dec_num_blocks 2 \
                --feature_dims 32
