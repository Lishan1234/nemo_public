#!/bin/sh
python $MOBINAS_CODE_ROOT/cache_profile/frame_dependency_analyzer.py \
                --dataset_dir $MOBINAS_DATA_ROOT/game-lol \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --video_name 240p_s0_d60_encoded.webm \
                --gop 120 \
                --chunk_idx 5 \
                --num_visualized_frames 10
