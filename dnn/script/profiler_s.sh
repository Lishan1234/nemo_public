#!/bin/bash
python $DNNDIR/tool/profiler_s.py \
                --model_dir $DATADIR/model \
                --snpe_dir /home/hyunho/snpe-1.32.0.555 \
                --num_blocks 8 \
                --num_filters 8 16 32 48 64 \
                --scale 4 \
                --nhwc 1 240 426 3 \
                --device_id a152b92a \
                --runtime GPU_FP16
