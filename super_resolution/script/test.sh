#!/bin/sh
python setup_videos.py --dataset starcraft --scale 4 --video_start 120 --lq_dnn EDSR_transpose_B8_F8 --hq_dnn EDSR_transpose_B8_F32
#python setup_videos.py --dataset movie --scale 4 --video_start 125 --lq_dnn EDSR_transpose_B8_F8 --hq_dnn EDSR_transpose_B8_F32
#python setup_videos.py --dataset basketball --scale 4 --video_start 1090 --lq_dnn EDSR_transpose_B8_F8 --hq_dnn EDSR_transpose_B8_F32

#python setup_graphs.py --dataset starcraft --scale 4 --video_start 120 --lq_dnn EDSR_transpose_B8_F8 --hq_dnn EDSR_transpose_B8_F32 --start_idx 0 --end_idx 500
#python setup_graphs.py --dataset movie --scale 4 --video_start 125 --lq_dnn EDSR_transpose_B8_F8 --hq_dnn EDSR_transpose_B8_F32
#python setup_graphs.py --dataset basketball --scale 4 --video_start 1090 --lq_dnn EDSR_transpose_B8_F8 --hq_dnn EDSR_transpose_B8_F32

#python setup_videos.py --dataset starcraft --scale 3 --video_start 120 --lq_dnn EDSR_transpose_B4_F8 --hq_dnn EDSR_transpose_B8_F32
#python setup_videos.py --dataset movie --scale 3 --video_start 125 --lq_dnn EDSR_transpose_B4_F8 --hq_dnn EDSR_transpose_B8_F32
#python setup_videos.py --dataset basketball --scale 3 --video_start 1090 --lq_dnn EDSR_transpose_B4_F8 --hq_dnn EDSR_transpose_B8_F32

#python setup_graphs.py --dataset starcraft --scale 3 --video_start 120 --lq_dnn EDSR_transpose_B4_F8 --hq_dnn EDSR_transpose_B8_F32
#python setup_graphs.py --dataset movie --scale 3 --video_start 125 --lq_dnn EDSR_transpose_B4_F8 --hq_dnn EDSR_transpose_B8_F32
#python setup_graphs.py --dataset basketball --scale 3 --video_start 1090 --lq_dnn EDSR_transpose_B4_F8 --hq_dnn EDSR_transpose_B8_F32

#python setup_videos.py --dataset starcraft --scale 2 --video_start 120 --lq_dnn EDSR_transpose_B4_F4 --hq_dnn EDSR_transpose_B8_F32
#python setup_videos.py --dataset movie --scale 2 --video_start 125 --lq_dnn EDSR_transpose_B4_F4 --hq_dnn EDSR_transpose_B8_F32
#python setup_videos.py --dataset basketball --scale 2 --video_start 1090 --lq_dnn EDSR_transpose_B4_F8 --hq_dnn EDSR_transpose_B8_F32

#python setup_graphs.py --dataset starcraft --scale 2 --video_start 120 --lq_dnn EDSR_transpose_B4_F4 --hq_dnn EDSR_transpose_B8_F32
#python setup_graphs.py --dataset movie --scale 2 --video_start 125 --lq_dnn EDSR_transpose_B4_F8 --hq_dnn EDSR_transpose_B8_F32
#python setup_graphs.py --dataset basketball --scale 2 --video_start 1090 --lq_dnn EDSR_transpose_B4_F8 --hq_dnn EDSR_transpose_B8_F32
