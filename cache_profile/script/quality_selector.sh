#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-g GPU_INDEX] [-c CONTENTS] [-i INDEXES] [-r RESOLUTIONS] [-d DEVICE_NAMES]

mandatory arguments:
-d DEVICE_NAMES               Specifies device_names

optional multiple arguments:
-c CONTENTS                 Specifies contents (e.g., product_review)
-r RESOLUTIONS              Specifies resolutions (e.g., 240)
-i INDEXES                  Specifies indexes (e.g., 0)

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
    fi
}

function _set_device_name(){
    device_name_arg=""
    for device_name in "${device_names[@]}"
    do
        device_name_arg+="${device_name} "
    done
}

[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts ":c:i:r:d:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        c) contents+=("$OPTARG");;
        i) indexes+=("$OPTARG");;
        r) resolutions+=("$OPTARG");;
        d) device_names+=("$OPTARG");;
        \?) exit 1;
    esac
done

if [ -z "${device_names+x}" ]; then
    echo "[ERROR] device_names is not set"
    exit 1;
fi

if [ -z "${contents+x}" ]; then
    contents=("product_review" "vlogs" "how_to" "skit" "game_play" "haul" "challenge" "education" "favorite" "unboxing")
fi

if [ -z "${resolutions+x}" ]; then
    resolutions=("240" "360" "480")
fi

if [ -z "${indexes+x}" ]; then
    indexes=("1" "2" "3")
fi


_set_conda
_set_device_name

for content in "${contents[@]}"
do
    for index in "${indexes[@]}"
    do
        for resolution in "${resolutions[@]}";
        do
            _set_bitrate ${resolution}
            python ${NEMO_ROOT}/cache_profile/quality_selector.py --data_dir ${NEMO_ROOT}/data --content ${content}${index} --video_name ${resolution}p_${bitrate}kbps_s0_d300.webm --device_name ${device_name_arg}
        done
    done
done
