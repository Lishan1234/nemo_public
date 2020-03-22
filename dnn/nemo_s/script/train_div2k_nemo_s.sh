#!/bin/bash
python $MOBINAS_CODE_ROOT/dnn/nemo_s/train_div2k_nemo_s.py \
                --lr_image_dir /ssd1/div2k/DIV2K_train_LR_bicubic/X4 \
                --hr_image_dir /ssd1/div2k/DIV2K_train_HR \
                --num_blocks 8 \
                --num_filters 32 \
                --scale 4 \
                --upsample_type deconv \
                --num_steps 1000 \
                --load_on_memory 
