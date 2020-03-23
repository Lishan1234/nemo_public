#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval01_computing_overhead.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content challenge education favorite game_play haul how_to product_review skit unboxing vlogs \
                --resolution 240 \
                --baseline_num_blocks 8 8 8 \
                --baseline_num_filters 9 21 32 \
                --threshold 0.5 \
                --aps_class nemo \
                --device_id 7b7f59d1
