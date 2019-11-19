#!/bin/bash
python $DNNDIR/trainer_s.py --dataset_dir $DATADIR/game-lol \
                --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
                --lr_video_name 240p_s0_d60.webm \
                --hr_video_name 960p_s0_d60.webm \
                --num_blocks 8 \
                --num_filters 64 \
                --load_on_memory 
