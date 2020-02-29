#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval03_02_c.py \
                --dataset_rootdir /ssd/nemo-mobicom \
                --content unboxing \
                --lr_resolution 240 \
                --hr_resolution 1080 \
                --num_blocks 8 \
                --num_filters 32 \
                --upsample_type deconv \
                --threshold 0.5 \
                --chunk_idx 10
