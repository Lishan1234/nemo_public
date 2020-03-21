#!/bin/sh
python $MOBINAS_CODE_ROOT/cache_profile/nemo_s/frame_dependency_analyzer.py \
                --dataset_dir $MOBINAS_DATA_ROOT/unboxing \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --lr_video_name 240p_s0_d300_encoded.webm \
                --hr_video_name 960p_240p_s0_d300.webm \
                --gop 120 \
                --num_blocks 8 \
                --num_filters 32 \
                --gop 120 \
                --threshold 0.5 \
                --mode nemo \
                --num_visualized_frames 30 

python $MOBINAS_CODE_ROOT/cache_profile/nemo_s/frame_dependency_analyzer.py \
                --dataset_dir $MOBINAS_DATA_ROOT/unboxing \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --lr_video_name 240p_s0_d300_encoded.webm \
                --hr_video_name 960p_240p_s0_d300.webm \
                --gop 120 \
                --num_blocks 8 \
                --num_filters 32 \
                --gop 120 \
                --threshold 0.5 \
                --mode uniform \
                --num_visualized_frames 30 

python $MOBINAS_CODE_ROOT/cache_profile/nemo_s/frame_dependency_analyzer.py \
                --dataset_dir $MOBINAS_DATA_ROOT/unboxing \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --lr_video_name 240p_s0_d300_encoded.webm \
                --hr_video_name 960p_240p_s0_d300.webm \
                --gop 120 \
                --num_blocks 8 \
                --num_filters 32 \
                --gop 120 \
                --threshold 0.5 \
                --mode random \
                --num_visualized_frames 30 
