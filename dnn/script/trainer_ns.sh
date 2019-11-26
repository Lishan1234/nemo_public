#!/bin/bash
python $DNNDIR/trainer_ns.py --dataset_dir $DATADIR/game-lol \
                --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
                --lr_video_name 240p_s0_d60.webm \
                --feature_video_name 240p_s0_d60_linear_0,100_encode.webm \
                --encode_model_name EDSR_ED_S_B8_F64_B1_F8_S4 \
                --num_blocks 8 \
                --num_filters 64 \
                --load_on_memory 
