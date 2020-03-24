#!/bin/bash
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval_youngmok.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content challenge education favorite game_play haul how_to product_review skit unboxing vlogs \
                --upsample_type deconv \
                --threshold 0.5 \
                --aps_class nemo \
                --device_id 7b7f59d1

python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval_youngmok.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content challenge education favorite game_play haul how_to product_review skit unboxing vlogs \
                --upsample_type deconv \
                --threshold 0.5 \
                --aps_class nemo \
                --device_id 10098e40 

python $MOBINAS_CODE_ROOT/evaluation/nemo_s/eval_youngmok.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content challenge education favorite game_play haul how_to product_review skit unboxing vlogs \
                --upsample_type deconv \
                --threshold 0.5 \
                --aps_class nemo \
                --device_id LMT605728961d9
