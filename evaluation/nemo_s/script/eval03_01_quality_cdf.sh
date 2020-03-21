#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval03_01_quality_cdf.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content challenge education favorite game_play haul how_to product_review skit unboxing vlogs \
                --lr_resolution 240 \
                --hr_resolution 1080 \
                --num_blocks 8 \
                --num_filters 32 \
                --upsample_type deconv \
                --threshold 0.5 
