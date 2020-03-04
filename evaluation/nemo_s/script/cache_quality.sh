#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 
conda activate nemo
python $MOBINAS_CODE_ROOT/evaluation/nemo_s/cache_quality.py \
                --dataset_rootdir $MOBINAS_DATA_ROOT \
                --content how_to \
                --vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \
                --lr_resolution 240 \
                --hr_resolution 1080 \
                --num_blocks 8 \
                --num_filters 32 \
                --upsample_type deconv \
                --threshold 0.5 \
                --aps_class nemo 
