#!/bin/bash

python $MOBINAS_CODE_ROOT/tool/encoder.py \
        --video_dir $MOBINAS_DATA_ROOT/game-lol/video \
        --ffmpeg_path /usr/bin/ffmpeg \
        --start_time 0  \
        --duration 60
