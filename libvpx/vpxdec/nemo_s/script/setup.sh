#!/bin/bash

python $MOBINAS_CODE_ROOT/libvpx/vpxdec/nemo_s/setup.py --dataset_dir $MOBINAS_DATA_ROOT/product_review \
    --lib_dir $MOBINAS_CODE_ROOT/libvpx/vpxdec/libs \
    --lr_video_name 240p_s0_d300_encoded.webm \
    --hr_video_name 960p_240p_s0_d300.webm \
    --upsample_type deconv \
    --num_filters 32 21 \
    --num_blocks 8 8 \
    --device_id 7b7f59d1 \
    --abi arm64-v8a \
    --aps_class nemo \
    --threshold 0.5 \
    --limit 240

python $MOBINAS_CODE_ROOT/libvpx/vpxdec/nemo_s/setup.py --dataset_dir $MOBINAS_DATA_ROOT/how_to \
    --lib_dir $MOBINAS_CODE_ROOT/libvpx/vpxdec/libs \
    --lr_video_name 240p_s0_d300_encoded.webm \
    --hr_video_name 960p_240p_s0_d300.webm \
    --upsample_type deconv \
    --num_filters 32 21 \
    --num_blocks 8 8 \
    --device_id 7b7f59d1 \
    --abi arm64-v8a \
    --aps_class nemo \
    --threshold 0.5 \
    --limit 240

python $MOBINAS_CODE_ROOT/libvpx/vpxdec/nemo_s/setup.py --dataset_dir $MOBINAS_DATA_ROOT/vlogs \
    --lib_dir $MOBINAS_CODE_ROOT/libvpx/vpxdec/libs \
    --lr_video_name 240p_s0_d300_encoded.webm \
    --hr_video_name 960p_240p_s0_d300.webm \
    --upsample_type deconv \
    --num_filters 32 21 \
    --num_blocks 8 8 \
    --device_id 7b7f59d1 \
    --abi arm64-v8a \
    --aps_class nemo \
    --threshold 0.5 \
    --limit 240
