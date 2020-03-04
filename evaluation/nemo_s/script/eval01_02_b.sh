#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval01_02_b.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content unboxing \
                --lr_resolution 240 \
                --hr_resolution 1080 \
                --num_blocks 8 \
                --num_filters 32 \
                --baseline_num_blocks 8 8 8 \
                --baseline_num_filters 9 21 32 \
                --upsample_type deconv \
                --threshold 0.5 \
                --aps_class nemo \
                --device_id 7b7f59d1
