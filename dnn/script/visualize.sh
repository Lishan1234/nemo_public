#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS] [-i INDEXES] [-q QUALITIES] [-r RESOLUTIONS] [-s SCALE]


mandatory arguments:
-c CONTENT                  Specifies content (e.g., product_review0)
-i INDEXES                  Specifies indexes (e.g., 0)
-q QUALITIY                 Specifies quality (e.g., low)
-r RESOLUTION               Specifies resolution (e.g., 240)
-s SCALE                    Specifies dnn scale (e.g., 4)

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

while getopts ":c:i:q:r:s:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        c) content="$OPTARG";;
        i) index="$OPTARG";;
        q) quality="$OPTARG";;
        r) resolution="$OPTARG";;
        s) scale=("$OPTARG");;
        \?) exit 1;
    esac
done

if [ -z "${content+x}" ]; then
    echo "[ERROR] content is not set"
    exit 1;
fi

if [ -z "${index+x}" ]; then
    echo "[ERROR] index is not set"
    exit 1;
fi

if [ -z "${quality+x}" ]; then
    echo "[ERROR] quality is not set"
    exit 1;
fi

if [ -z "${resolution+x}" ]; then
    echo "[ERROR] resolution is not set"
    exit 1;
fi

if [ -z "${scale+x}" ]; then
    echo "[ERROR] scale is not set"
    exit 1;
fi

_set_conda

_set_bitrate ${resolution}
_set_num_blocks ${resolution} ${quality}
_set_num_filters ${resolution} ${quality}

python ${NEMO_ROOT}/dnn/visualize.py --dataset_dir ${NEMO_ROOT}/data/${content}${index}/ --video_name ${resolution}p_${bitrate}kbps_s0_d300.webm  --num_blocks ${num_blocks} --num_filters ${num_filters} --scale ${scale}
