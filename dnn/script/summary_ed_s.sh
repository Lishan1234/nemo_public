#!/bin/bash
python $DNNDIR/tool/summary_ed_s.py --rootdir $DATADIR \
                --dataset_name game-lol \
                --lr_video_name 240p_s0_d60.webm \
                --train_video_name 240p_s0_d60.webm \
                --hr_video_name 960p_s0_d60.webm \
                --device_id a152b92a \
                --runtime GPU_FP16 \
                --enc_num_blocks 8 \
                --enc_num_filters 64 \
                --dec_num_blocks 1 \
                --dec_num_filters 8 \
                --min_percentile 0 \
                --max_percentile 100 \
                --bitrate 400
