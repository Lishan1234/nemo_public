#!/bin/bash

python $MOBINAS_CODE_ROOT/tool/encoder.py \
        --video_dir $MOBINAS_DATA_ROOT/product_review/video \
        --ffmpeg_path /usr/bin/ffmpeg \
        --start_time 0 
