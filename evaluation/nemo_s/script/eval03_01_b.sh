#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval03_01_b.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content unboxing \
                --upsample_type deconv \
                --threshold 0.5 \
                --aps_class nemo \
                --device_id 7b7f59d1
