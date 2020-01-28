#!/bin/bash
python $MOBINAS_CODE_ROOT/cache_profile/nemo_s/anchor_point_selector_nemo_s.py \
                --dataset_dir $MOBINAS_DATA_ROOT/game-lol \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --lr_video_name 240p_s0_d60_encoded.webm \
                --hr_video_name 960p_s0_d60.webm \
                --num_blocks 8 \
                --num_filters 48 \
                --gop 120 \
                --threshold 0.2 \
                --mode nemo \
                --chunk_idx 5
