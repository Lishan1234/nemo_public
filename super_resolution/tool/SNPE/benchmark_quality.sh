#!/bin/bash
python benchmark_quality.py --num_blocks 4 --num_filters 32 --upsample_type transpose --snpe_project_root ../../../snpe --train_data news --lr 240
