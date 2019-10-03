#!/bin/sh
python cache_erosion_analyzer.py --cra_content_dir /home/hyunho/MobiNAS/super_resolution/data/movie/result --cra_input_video_name 270p_512k_60sec_125st.webm --cra_dnn_video_name 1080p_270p_60sec_125st_EDSR_transpose_B8_F32_S4.webm --cra_compare_video_name 1080p_lossless_60sec_125st.webm --cra_total_frames 30
