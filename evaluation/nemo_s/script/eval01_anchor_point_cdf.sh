#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval01_anchor_point_cdf.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content challenge education favorite game_play haul how_to product_review skit unboxing vlogs \
                --resolution 240 \
                --aps_class nemo \
                --threshold 0.5 \
                --device_id 7b7f59d1
