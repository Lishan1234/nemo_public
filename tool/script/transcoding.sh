#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS] [i INDEXES]

optional arguments:
-c CONTENTS                 Specifies contents (e.g., product_review)
-i INDEXES                  Specifies indexes (e.g., 0)

EOF
}

function _transcode()
{
    #cut and encode (2160p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1$2/video --input_video_path $NEMO_ROOT/data/video/$1$2.webm --start 0 --duration 300 --bitrate 12000 --mode cut_and_resize_and_encode --output_width 3840 --output_height 2160

    #encode (240p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1$2/video --input_video_path $NEMO_ROOT/data/$1$2/video/2160p_12000kbps_s0_d300.webm --bitrate 512 --output_width 426 --output_height 240 --start 0 --duration 300 --mode resize_and_encode

    #encode (360p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1$2/video --input_video_path $NEMO_ROOT/data/$1$2/video/2160p_12000kbps_s0_d300.webm --bitrate 1024 --output_width 640 --output_height 360 --start 0 --duration 300 --mode resize_and_encode

    #encode (480p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1$2/video --input_video_path $NEMO_ROOT/data/$1$2/video/2160p_12000kbps_s0_d300.webm --bitrate 1600 --output_width 854 --output_height 480 --start 0 --duration 300 --mode resize_and_encode

    #encode (720p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1$2/video --input_video_path $NEMO_ROOT/data/$1$2/video/2160p_12000kbps_s0_d300.webm --bitrate 2640 --output_width 1280 --output_height 720 --start 0 --duration 300 --mode resize_and_encode

    #encode (1080p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1$2/video --input_video_path $NEMO_ROOT/data/$1$2/video/2160p_12000kbps_s0_d300.webm --bitrate 4400 --output_width 1920 --output_height 1080 --start 0 --duration 300 --mode resize_and_encode

    #encode (1440p)
    python $NEMO_ROOT/tool/encoder.py --output_video_dir $NEMO_ROOT/data/$1$2/video --input_video_path $NEMO_ROOT/data/$1$2/video/2160p_12000kbps_s0_d300.webm --bitrate 8000 --output_width 2560 --output_height 1440 --start 0 --duration 300 --mode resize_and_encode
}

[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts "c:i:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        i) indexes+=("$OPTARG");;
        c) contents+=("$OPTARG");;
        \?) exit 1;
    esac
done

if [ -z "${contents}" ] ; then
    echo "[ERROR] contents is not set"
    exit 1;
fi

if [ -z "${indexes+x}" ]; then
    indexes=("1" "2" "3")
fi

if [ -z "${contents+x}" ]; then
    contents=("product_review" "how_to" "vlogs" "skit" "game_play" "haul" "challenge" "education" "favorite" "unboxing")
fi

for content in "${contents[@]}"; do
    for index in "${indexes[@]}"
    do
        _transcode $content $index
    done
done
