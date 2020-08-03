#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS] [-i INDEXES] [-r RESOLUTIONS] [-o OUTPUT_RESOLUTION]

mandatory arguments:
-c CONTENTS                 Specifies contents (e.g., product_review)

optional multiple arguments:
-i INDEXES                  Specifies indexes (e.g., 0)
-r RESOLUTIONS              Specifies resolutions (e.g., 240)
-o OUTPUT_RESOLUTION        Specifies output resolution  (e.g., 1080)

EOF
}

function _set_conda(){
    source ~/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate nemo_py3.6
}

function _set_bitrate(){
    if [ "$1" == 240 ];then
        bitrate=512
    elif [ "$1" == 360 ];then
        bitrate=1024
    elif [ "$1" == 480 ];then
        bitrate=1600
    elif [ "$1" == 720 ];then
        bitrate=2640
    fi
}

function _set_output_size(){
    if [ "$1" == 1080 ];then
        output_width=1920
        output_height=1080
    elif [ "$1" == 1440 ];then
        output_width=2560
        output_height=1440
    elif [ "$1" == 2160 ];then
        output_width=3840
        output_height=2160
    fi
}

[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts ":c:i:r:o:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        c) contents+=("$OPTARG");;
        i) indexes+=("$OPTARG");;
        r) resolutions+=("$OPTARG");;
        o) output_resolution="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${contents+x}" ]; then
    echo "[ERROR] contents is not set"
    exit 1;
fi

if [ -z "${resolutions+x}" ]; then
    resolutions=("1080")
fi

if [ -z "${indexes+x}" ]; then
    indexes=("1" "2" "3")
fi

if [ -z "${output_resolution+x}" ]; then
    output_resolution=1080
fi

_set_conda
_set_output_size ${output_resolution}

for content in "${contents[@]}"
do
    for index in "${indexes[@]}"
    do
        for resolution in "${resolutions[@]}";
        do
            _set_bitrate ${resolution}
            python ${NEMO_ROOT}/cache_profile/measure_bilinear_quality.py --data_dir ${NEMO_ROOT}/data --content ${content}${index} --lr_video_name ${resolution}p_${bitrate}kbps_s0_d300.webm --hr_video_name 2160p_12000kbps_s0_d300.webm --output_width ${output_width} --output_height ${output_height}
        done
    done
done
