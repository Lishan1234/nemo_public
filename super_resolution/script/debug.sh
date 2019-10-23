#!/bin/sh
python setup_debug.py --dataset movie --scale 4 --video_start 125 --lq_dnn EDSR_transpose_B8_F8 --hq_dnn EDSR_transpose_B8_F32 --cache_only
