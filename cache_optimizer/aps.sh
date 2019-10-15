#!/bin/sh
python anchor_point_selector.py --vpxdec_path /home/hyunho/git/libvpx/vpxdec --content_dir /home/hyunho/MobiNAS/super_resolution/data/movie --input_video_name 270p_512k_60sec_125st.webm --dnn_video_name 1080p_270p_60sec_125st_EDSR_transpose_B8_F32_S4.webm --compare_video_name 1080p_lossless_60sec_125st.webm --num_frames 30
