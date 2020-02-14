#!/bin/bash
python $MOBINAS_CODE_ROOT/dnn/vgg16/run_snpe_vgg16.py \
                    --log_dir $MOBINAS_DATA_ROOT/vgg16  \
                    --device_id a152b92a \
                    --runtime GPU_FP16 
