#!/bin/bash
python $DNNDIR/tool/decode_ed_s.py --dataset_dir $DATADIR/game-lol \
                --ffmpeg_path $FFMPEGPATH \
                --start_time 0 \
                --duration 60 \
                --filter_type none \
                --input_resolution 270 \
                --target_resolution 1080 \
                --load_on_memory \
                --enc_num_filters 64 \
                --enc_num_blocks 8 \
                --dec_num_filters 8 \
                --dec_num_blocks 1 \
                --checkpoint_dir /ssd1/dataset-sigcomm2020/game-lol/checkpoint/270p_s0_d60_encoded.webm.uniform_1.00/EDSR_ED_S_B8_F64_B1_F8_S4 \
                --feature_image_dir /ssd1/dataset-sigcomm2020/game-lol/image/270p_s0_d60_encoded.webm/encode/270p_feature_s0_e60_crf25
                #--feature_image_dir /ssd1/dataset-sigcomm2020/game-lol/image/270p_s0_d60_encoded.webm/encode 
