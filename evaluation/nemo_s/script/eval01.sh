#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval01.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content game-lol \
                --lr_video_name 240p_s0_d60_encoded.webm \
                --hr_video_name 1080p_s0_d60.webm \
                --num_blocks 8 \
                --num_filters 48 \
                --threshold 0.2 \
                --aps_class uniform 
