#!/bin/bash

#cut & lossless encode (1080p)
#python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1/video --input_video_path $NEMO_ROOT/data/video/$1.webm --input_height 2160 --start 0 --duration 300 --mode cut
#python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1/video --input_video_path $NEMO_ROOT/data/$1/video/2160p_s0_d300.webm --bitrate 0 --input_height 2160 --output_width 1920 --output_height 1080 --start 0 --duration 300 --mode encode

#encode (240p)
python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1/video --input_video_path $NEMO_ROOT/data/$1/video/1080p_s0_d300.webm --bitrate 512 --input_height 1080 --output_width 426 --output_height 240 --start 0 --duration 300 --mode encode

#encode (360p)
#python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1/video --input_video_path $NEMO_ROOT/data/$1/video/1080p_s0_d300.webm --bitrate 1024 --input_height 1080 --output_width 640 --output_height 360 --start 0 --duration 300 --mode encode

#encode (480p)
#python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1/video --input_video_path $NEMO_ROOT/data/$1/video/1080p_s0_d300.webm --bitrate 1600 --input_height 1080 --output_width 854 --output_height 480 --start 0 --duration 300 --mode encode

#encode (720p)
#python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1/video --input_video_path $NEMO_ROOT/data/$1/video/1080p_s0_d300.webm --bitrate 2640 --input_height 1080 --output_width 1280 --output_height 720 --start 0 --duration 300 --mode encode

#encode (1080p)
#python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1/video --input_video_path $NEMO_ROOT/data/$1/video/1080p_s0_d300.webm --bitrate 4400 --input_height 1080 --output_width 1920 --output_height 1080 --start 0 --duration 300 --mode encode
