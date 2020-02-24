#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/cache_quality.py \
                --dataset_dir $MOBINAS_DATA_ROOT/how_to \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --lr_video_name 240p_s0_d300_encoded.webm \
                --hr_video_name 1080p_s0_d300.webm \
                --num_blocks 8 \
                --num_filters 32 \
                --upsample_type deconv \
                --threshold 0.5 \
                --aps_class nemo \
                --setup_image \
                --remove_image
