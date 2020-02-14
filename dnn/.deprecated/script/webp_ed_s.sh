#!/bin/bash
python $DNNDIR/tool/webp_ed_s.py --dataset_dir $DATADIR/game-lol \
                --webp_dir $HOME/libwebp-1.0.3-linux-x86-64 \
                --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
                --lr_video_name 240p_s0_d60.webm \
                --hr_video_name 960p_s0_d60.webm \
                --train_video_name 240p_s0_d60.webm \
                --enc_num_blocks 8 \
                --enc_num_filters 64 \
                --dec_num_blocks 8 \
                --dec_num_filters 8 16 32 48 64 \
                --min_percentile 0 \
                --max_percentile 100 
