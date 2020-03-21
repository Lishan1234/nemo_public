#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval_section5_b.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content unboxing \
                --lr_resolution 240 \
                --hr_resolution 960 \
                --num_blocks 8 \
                --num_filters 32 \
                --upsample_type deconv 
