#!/bin/bash
python $MOBINAS_CODE_ROOT/dnn/alexnet/run_snpe_alexnet.py \
                    --log_dir $MOBINAS_DATA_ROOT/alexnet \
                    --device_id a152b92a \
                    --runtime GPU_FP16 
