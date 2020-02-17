#!/bin/bash
python $MOBINAS_CODE_ROOT/dnn/nemo_s_y/run_snpe_nemo_s.py \
                --dataset_dir $MOBINAS_DATA_ROOT/game-lol \
                --lr_video_name 240p_s0_d60_encoded.webm \
                --hr_video_name 1080p_s0_d60.webm \
                --num_blocks 8 \
                --num_filters 48 \
                --upsample_type deconv \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --device_id 7b7f59d1 \
                --runtime GPU_FP16 
