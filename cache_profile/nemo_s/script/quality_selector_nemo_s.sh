#!/bin/bash
python $MOBINAS_CODE_ROOT/cache_profile/nemo_s/quality_selector_nemo_s.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --reference_content unboxing \
                --content unboxing \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --lr_resolution 240 \
                --hr_resolution 1080 \
                --num_blocks 8 8 8 \
                --num_filters 9 21 32 \
                --gop 120 \
                --threshold 0.5 \
                --device_id 7b7f59d1
