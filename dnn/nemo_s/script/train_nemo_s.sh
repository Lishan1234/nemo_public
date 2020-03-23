#!/bin/bash
python $MOBINAS_CODE_ROOT/dnn/nemo_s/train_nemo_s.py --dataset_dir $MOBINAS_DATA_ROOT/unboxing \
                --ffmpeg_path $MOBINAS_FFMPEG_PATH \
                --lr_video_name 240p_s0_d300_encoded.webm \
                --hr_video_name 960p_240p_s0_d300.webm \
                --num_blocks 8 \
                --num_filters 32 \
                --upsample_type deconv \
                --num_steps 1000 \
                --load_on_memory \
                --div2k_checkpoint_dir $MOBINAS_DATA_ROOT/div2k
