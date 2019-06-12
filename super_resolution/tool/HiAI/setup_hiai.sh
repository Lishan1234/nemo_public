#!/bin/bash
python setup_hiai.py --num_blocks 4 --num_filters 32 --upsample_type transpose --hwc 240,426,3 --train_data news --data_type 60_0.5 --lr 240 --snpe_project_root '../../../snpe' --snpe_tensorflow_root '../../../../tensorflow'
