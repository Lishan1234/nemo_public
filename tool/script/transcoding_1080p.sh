#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS]

mandatory arguments:
-c CONTENTS                 Specifies contents (e.g., product_review0)

EOF
}

function _transcode_1080p()
{
    #cut & lossless encode (1080p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$content/video --input_video_path $NEMO_ROOT/data/video/$content.webm --start 0 --duration 300 --mode cut

    #encode (240p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$content/video --input_video_path $NEMO_ROOT/data/$content/video/2160p_s0_d300.webm --bitrate 512 --output_width 426 --output_height 240 --start 0 --duration 300 --mode encode

    #encode (360p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$content/video --input_video_path $NEMO_ROOT/data/$content/video/2160p_s0_d300.webm --bitrate 1024 --output_width 640 --output_height 360 --start 0 --duration 300 --mode encode

    #encode (480p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$content/video --input_video_path $NEMO_ROOT/data/$content/video/2160p_s0_d300.webm --bitrate 1600 --output_width 854 --output_height 480 --start 0 --duration 300 --mode encode

    #encode (720p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$content/video --input_video_path $NEMO_ROOT/data/$content/video/2160p_s0_d300.webm --bitrate 2640 --output_width 1280 --output_height 720 --start 0 --duration 300 --mode encode

    #encode (1080p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$content/video --input_video_path $NEMO_ROOT/data/$content/video/2160p_s0_d300.webm --bitrate 4400 --output_width 1920 --output_height 1080 --start 0 --duration 300 --mode encode
}

[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts "c:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        c) contents+=("$OPTARG");;
        \?) exit 1;
    esac
done

for content in "${contents[@]}"; do
    _transcode_1080p
done
