#!/bin/bash
python $MOBINAS_CODE_ROOT/cache_profile/nemo_s/anchor_point_selector_nemo_s.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content unboxing \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --lr_video_name 240p_s0_d300_encoded.webm \
                --hr_video_name 1080p_s0_d300.webm \
                --num_blocks 8 \
                --num_filters 32 \
                --gop 120 \
                --mode nemo \
                --task profile \
                --chunk_idx 19


