#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval01_latency.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content unboxing \
                --resolution 240 \
                --threshold 0.5 \
                --aps_class nemo \
                --device_id 7b7f59d1
