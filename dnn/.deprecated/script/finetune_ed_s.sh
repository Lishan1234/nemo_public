#!/bin/bash
python $DNNDIR/finetune_ed_s.py --dataset_dir $DATADIR/game-lol \
                --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
                --lr_video_name 240p_s0_d60.webm \
                --hr_video_name 960p_s0_d60.webm \
                --enc_num_blocks 8 \
                --enc_num_filters 64 \
                --dec_num_blocks 1 \
                --dec_num_filters 8 \
                --bitrate 200 \
                --min_percentile 0 \
                --max_percentile 100 \
                --learning_rate 1e-4 