#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS] [-i INDEXES] [-q QUALITIES] [-r RESOLUTIONS] [-d DEVICE] [-t TRAIN_TYPE]

mandatory arguments:
-c CONTENTS                 Specifies contents (e.g., product_review)
-d DEVICE_ID                Specifies device id
-t TRAIN_TYPE               Specifies train_type (e.g., finetune_video)

optional multiple arguments:
-i INDEXES                  Specifies indexes (e.g., 0)
-q QUALITIES                Specifies qualities (e.g., low)
-r RESOLUTIONS              Specifies resolutions (e.g., 240)

EOF
}

function _set_conda(){
    source ~/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate nemo_py3.4
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

while getopts ":c:i:q:r:d:t:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        c) contents+=("$OPTARG");;
        i) indexes+=("$OPTARG");;
        q) qualities+=("$OPTARG");;
        r) resolutions+=("$OPTARG");;
        d) device_id="$OPTARG";;
        t) train_type="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${contents+x}" ] || [ -z "${device_id+x}" ] || [ -z "${train_type+x}" ]; then
    echo "[ERROR] contents and device_id and train_type must be set"
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
                python ${NEMO_ROOT}/tool/copy_data_to_mobile.py --data_dir ${NEMO_ROOT}/data --content ${content}${index} --lr_video_name ${resolution}p_${bitrate}kbps_s0_d300.webm --hr_video_name 2160p_s0_d300.webm --num_blocks ${num_blocks} --num_filters ${num_filters} --device_id=${device_id} --train_type=${train_type}
            done
        done
    done
done
