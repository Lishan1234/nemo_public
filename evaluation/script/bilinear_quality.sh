#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 
conda activate mobinas

python $MOBINAS_CODE_ROOT/evaluation/bilinear_quality.py \
    --dataset_rootdir $MOBINAS_DATA_ROOT \
    --content game_play_1 \
    --lr_resolution 240 \
    --hr_resolution 1080 \
    --ffmpeg_file /usr/bin/ffmpeg \
    --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec

python $MOBINAS_CODE_ROOT/evaluation/bilinear_quality.py \
    --dataset_rootdir $MOBINAS_DATA_ROOT \
    --content game_play_1 \
    --lr_resolution 360 \
    --hr_resolution 1080 \
    --ffmpeg_file /usr/bin/ffmpeg \
    --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec

python $MOBINAS_CODE_ROOT/evaluation/bilinear_quality.py \
    --dataset_rootdir $MOBINAS_DATA_ROOT \
    --content game_play_1 \
    --lr_resolution 480 \
    --hr_resolution 1080 \
    --ffmpeg_file /usr/bin/ffmpeg \
    --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec

python $MOBINAS_CODE_ROOT/evaluation/bilinear_quality.py \
    --dataset_rootdir $MOBINAS_DATA_ROOT \
    --content game_play_1 \
    --lr_resolution 720 \
    --hr_resolution 1080 \
    --ffmpeg_file /usr/bin/ffmpeg \
    --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec

python $MOBINAS_CODE_ROOT/evaluation/bilinear_quality.py \
    --dataset_rootdir $MOBINAS_DATA_ROOT \
    --content game_play_1 \
    --lr_resolution 1080 \
    --hr_resolution 1080 \
    --ffmpeg_file /usr/bin/ffmpeg \
    --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec
