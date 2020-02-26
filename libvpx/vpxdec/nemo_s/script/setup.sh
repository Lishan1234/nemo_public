#!/bin/bash

python $MOBINAS_CODE_ROOT/libvpx/vpxdec/nemo_s/setup.py \
    --dataset_rootdir $MOBINAS_DATA_ROOT \
    --content challenge education favorite game_play haul how_to product_review skit unboxing vlogs \
    --lib_dir $MOBINAS_CODE_ROOT/libvpx/vpxdec/libs \
    --lr_resolution 360 \
    --hr_resolution 1080 \
    --upsample_type deconv \
    --num_filters 8 \
    --num_blocks 8 \
    --device_id 7b7f59d1 \
    --abi arm64-v8a \
    --aps_class nemo \
    --threshold 0.5 
    #--content challenge education favorite game_play haul how_to product_review skit unboxing vlogs \
