#!/bin/bash
python $DNNDIR/tool/summary_s.py --rootdir $DATADIR \
                --dataset_name game-lol \
                --lr_video_name 240p_s0_d60.webm \
                --hr_video_name 960p_s0_d60.webm \
                --bitrate 0 200 400\
                --num_blocks 8 \
                --num_filters 8 \
                --device_id a152b92a \
                --runtime GPU_FP16
