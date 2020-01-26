#!/bin/bash
python $MOBINAS_CODE_ROOT/dnn/nemo_s/run_snpe_nemo_s.py \
                --dataset_dir $MOBINAS_DATA_ROOT/game-lol \
                --lr_video_name 240p_s0_d60_encoded.webm \
                --hr_video_name 960p_s0_d60.webm \
                --num_blocks 8 \
                --num_filters 9 \
                --device_id a152b92a \
                --runtime GPU_FP16 
