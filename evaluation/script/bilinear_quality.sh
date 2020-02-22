#!/bin/bash

python $MOBINAS_CODE_ROOT/evaluation/bilinear_quality.py \
    --dataset_rootdir $MOBINAS_DATA_ROOT \
    --content education \
    --lr_resolution 240 \
    --hr_resolution 1080 \
    --ffmpeg_file /usr/bin/ffmpeg \
    --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec
