#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval01_multiple_devices.py \
                --dataset_backup_rootdir /ssd2/nemo-mobicom-backup \
                --dataset_rootdir /ssd2/nemo-mobicom \
                --content unboxing \
                --resolution 240 \
                --threshold 0.5 \
                --aps_class nemo 
