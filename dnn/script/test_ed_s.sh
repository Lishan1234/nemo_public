#!/bin/bash
python $DNNDIR/tester_ed_s.py --dataset_dir $DATADIR/game-lol \
                --ffmpeg_path $FFMPEGPATH \
                --start_time 0 \
                --duration 60 \
                --input_resolution 270 \
                --target_resolution 1080 \
                --load_on_memory \
                --enc_num_filters 64 \
                --enc_num_blocks 8 \
                --dec_num_filters 8 \
                --dec_num_blocks 1 \
