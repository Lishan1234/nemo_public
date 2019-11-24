#!/bin/bash
python $DNNDIR/trainer_edd_s.py --dataset_dir $DATADIR/game-lol \
                --ffmpeg_path /home/hyunho/src/ffmpeg/ffmpeg \
                --lr_video_name 240p_s0_d60.webm \
                --hr_video_name 960p_s0_d60.webm \
                --enc_num_blocks 8 \
                --enc_num_filters 64 \
                --dec_lr_num_blocks 1 \
                --dec_lr_num_filters 8 \
                --dec_sr_num_blocks 1 \
                --dec_sr_num_filters 8 \
                --load_on_memory \
                --loss_type joint \
                --lr_loss_weight 0.1 \
                --sr_loss_weight 1
