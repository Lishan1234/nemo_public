#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval01_02_a.py \
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

python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval01_02_a.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content unboxing \
                --lr_resolution 360 \
                --hr_resolution 1080 \
                --num_blocks 8 \
                --num_filters 29 \
                --baseline_num_blocks 8 8 8 \
                --baseline_num_filters 8 18 29 \
                --upsample_type deconv \
                --threshold 0.5 \
                --aps_class nemo \
                --device_id 7b7f59d1

python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval01_02_a.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content unboxing \
                --lr_resolution 480 \
                --hr_resolution 1080 \
                --num_blocks 8 \
                --num_filters 18 \
                --baseline_num_blocks 8 8 8 \
                --baseline_num_filters 4 9 18 \
                --upsample_type deconv \
                --threshold 0.5 \
                --aps_class nemo \
                --device_id 7b7f59d1
