#!/bin/bash
python $MOBINAS_CODE_ROOT/dnn/nas_s/visualize_nas_s.py \
                --dataset_dir $MOBINAS_DATA_ROOT/game-lol \
                --lr_video_name 240p_s0_d60_encoded.webm \
                --hr_video_name 960p_s0_d60.webm \
                --num_blocks 2 \
                --num_filters 64 \
                --upsample_type subpixel 
