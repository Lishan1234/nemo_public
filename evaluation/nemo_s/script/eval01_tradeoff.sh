#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval01_tradeoff.py \
                --dataset_rootdir /ssd2/nemo-mobicom-backup \
                --content unboxing \
                --lr_resolution 240 \
                --hr_resolution 1080 \
                --num_blocks 8 \
                --num_filters 32 \
                --upsample_type deconv \
                --threshold 0.5 0.75 1.0 1.25 1.5 \
                --aps_class nemo \
                --device_id 7b7f59d1
