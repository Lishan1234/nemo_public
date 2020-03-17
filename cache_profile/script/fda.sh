#!/bin/sh
python $MOBINAS_CODE_ROOT/cache_profile/frame_dependency_analyzer.py \
                --dataset_dir $MOBINAS_DATA_ROOT/unboxing \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --video_name 240p_s0_d300_encoded.webm \
                --gop 120 \
                --num_visualized_frames 30 \
                --chunk_idx 0
