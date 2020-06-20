#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-g GPU_INDEX] [-c CONTENTS] [-i INDEXES] [-q QUALITIES] [-r RESOLUTIONS]

mandatory arguments:
 -g GPU_INDEX                Specifies GPU index to use
 -c CONTENT                  Specifies content (e.g., product_review0)
 -i INDEXES                  Specifies indexes (e.g., 0)
 -q QUALITIY                 Specifies quality (e.g., low)
 -r RESOLUTION               Specifies resolution (e.g., 240)

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

function _set_num_blocks(){
    if [ "$1" == 240 ];then
        if [ "$2" == "low" ];then
            num_blocks=8
        elif [ "$2" == "medium" ];then
            num_blocks=8
        elif [ "$2" == "high" ];then
            num_blocks=8
        fi
    elif [ "$1" == 360 ];then
        if [ "$2" == "low" ];then
            num_blocks=4
        elif [ "$2" == "medium" ];then
            num_blocks=4
        elif [ "$2" == "high" ];then
            num_blocks=4
        fi
    elif [ "$1" == 480 ];then
        if [ "$2" == "low" ];then
            num_blocks=4
        elif [ "$2" == "medium" ];then
            num_blocks=4
        elif [ "$2" == "high" ];then
            num_blocks=4
        fi
    fi
}

function _set_num_filters(){
    if [ "$1" == 240 ];then
        if [ "$2" == "low" ];then
            num_filters=9
        elif [ "$2" == "medium" ];then
            num_filters=21
        elif [ "$2" == "high" ];then
            num_filters=32
        fi
    elif [ "$1" == 360 ];then
        if [ "$2" == "low" ];then
            num_filters=8
        elif [ "$2" == "medium" ];then
            num_filters=18
        elif [ "$2" == "high" ];then
            num_filters=29
        fi
    elif [ "$1" == 480 ];then
        if [ "$2" == "low" ];then
            num_filters=4
        elif [ "$2" == "medium" ];then
            num_filters=9
        elif [ "$2" == "high" ];then
            num_filters=18
        fi
    fi
}

[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts ":g:c:i:q:r:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        g) gpu_index="$OPTARG";;
        c) content="$OPTARG";;
        i) index="$OPTARG";;
        q) quality="$OPTARG";;
        r) resolution="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${gpu_index+x}" ] || [ -z "${content+x}" ] || [ -z "${index+x}" ] || [ -z "${resolution}" ] || [ -z "${quality}" ]; then
    echo "[ERROR] gpu_index and content and resolution and quality must be set"
    exit 1;
fi

_set_conda

_set_bitrate ${resolution}
_set_num_blocks ${resolution} ${quality}
_set_num_filters ${resolution} ${quality}

CUDA_VISIBLE_DEVICES=${gpu_index} python ${NEMO_ROOT}/dnn/visualize.py --dataset_dir ${NEMO_ROOT}/data/${content}${index}/ --lr_video_name ${resolution}p_${bitrate}kbps_s0_d300.webm --hr_video_name 1080p_s0_d300.webm --num_blocks ${num_blocks} --num_filters ${num_filters}
