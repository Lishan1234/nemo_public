#!/bin/bash
python $MOBINAS_CODE_ROOT/dnn/nemo_s/run_tf_nemo_s.py \
                --dataset_dir $MOBINAS_DATA_ROOT/game-lol \
                --ffmpeg_path $MOBINAS_FFMPEG_PATH \
                --lr_video_name 240p_s0_d60_encoded.webm \
                --hr_video_name 1080p_s0_d60.webm \
                --num_blocks 8 \
                --num_filters 48 \
                --upsample_type deconv
