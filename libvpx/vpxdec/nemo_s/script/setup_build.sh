#!/bin/bash

python $MOBINAS_CODE_ROOT/libvpx/vpxdec/nemo_s/setup_build.py --video_dir $MOBINAS_DATA_ROOT/game-lol/video/ \
    --lib_dir $MOBINAS_CODE_ROOT/libvpx/vpxdec/libs \
    --video_name 240p_s0_d60_encoded.webm \
    --abi arm64-v8a \
    --device_id a152b92a 
