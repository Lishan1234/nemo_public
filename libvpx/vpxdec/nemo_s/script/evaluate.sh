#!/bin/bash

#TODO: add sleep as 30 minutes for a full evaluation
python $MOBINAS_CODE_ROOT/libvpx/vpxdec/nemo_s/evaluate.py --dataset_dir $MOBINAS_DATA_ROOT/how_to \
    --lr_video_name 480p_s0_d300_encoded.webm \
    --hr_video_name 1080p_s0_d300.webm \
    --upsample_type deconv \
    --num_filters 18 \
    --num_blocks 8 \
    --device_id a152b92a \
    --aps_class nemo \
    --threshold 0.5
