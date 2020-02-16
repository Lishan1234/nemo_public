#!/bin/bash
python $MOBINAS_CODE_ROOT/dnn/nas_s_y/train_nas_s.py --dataset_dir $MOBINAS_DATA_ROOT/game-lol \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --lr_video_name 240p_s0_d60_encoded.webm \
                --hr_video_name 960p_s0_d60.webm \
                --num_blocks 8 \
                --num_filters 48 \
                --num_steps 100000 \
                --load_on_memory 
