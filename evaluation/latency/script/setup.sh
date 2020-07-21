#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS] [-i INDEXES] [-q QUALITIES] [-r RESOLUTIONS] [-a ALGORITHM] [-d DEVICE_ID]

mandatory arguments:
-c CONTENTS                 Specifies contents (e.g., product_review)
-a ALGORITHM                Specifies algorithm (e.g., nemo)
-d DEVICE_ID                Specifies a device id

optional multiple arguments:
-i INDEXES                  Specifies indexes (e.g., 0)
-q QUALITIES                Specifies qualities (e.g., low)
-r RESOLUTIONS              Specifies resolutions (e.g., 240)

EOF
}

function _set_conda(){
    source ~/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate nemo_py3.5
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

while getopts ":c:i:q:r:t:a:d:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        a) algorithm="$OPTARG";;
        c) contents+=("$OPTARG");;
        i) indexes+=("$OPTARG");;
        q) qualities+=("$OPTARG");;
        r) resolutions+=("$OPTARG");;
        d) device_id+="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${contents+x}" ]; then
    echo "[ERROR] contents is not set"
    exit 1;
fi

if [ -z "${algorithm+x}" ]; then
    echo "[ERROR] algorithm is not set"
    exit 1;
fi

if [ -z "${device_id+x}" ]; then
    echo "[ERROR] device_id is not set"
    exit 1;
fi

if [ -z "${qualities+x}" ]; then
    qualities=("low" "medium" "high")
fi

if [ -z "${resolutions+x}" ]; then
    resolutions=("240" "360" "480")
fi

if [ -z "${indexes+x}" ]; then
    indexes=("1" "2" "3")
fi

_set_conda

for content in "${contents[@]}"
do
    for index in "${indexes[@]}"
    do
        for quality in "${qualities[@]}"
        do
            for resolution in "${resolutions[@]}";
            do
                _set_bitrate ${resolution}
                _set_num_blocks ${resolution} ${quality}
                _set_num_filters ${resolution} ${quality}
                CUDA_VISIBLE_DEVICES=${gpu_index} python ${NEMO_ROOT}/evaluation/latency/setup.py --data_dir ${NEMO_ROOT}/data --content ${content}${index} --video_name ${resolution}p_${bitrate}kbps_s0_d300.webm --lib_dir ${NEMO_ROOT}/evaluation/latency/libs/arm64-v8a --num_blocks ${num_blocks} --num_filters ${num_filters} --algorithm=${algorithm} --device_id=${device_id}
            done
        done
    done
done
