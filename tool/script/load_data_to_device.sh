#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS] [-i INDEXES] [-q QUALITIES] [-r RESOLUTIONS] [-d DEVICE_ID] [-e DEVICE_DATA_DIR]

mandatory arguments:
-d DEVICE_ID                Specifies a device id
-e DEVICE_DATA_DIR          Specifies device data directory (e.g., /sdcard/NEMO)

optional multiple arguments:
-c CONTENTS                 Specifies contents (e.g., product_review)
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
            num_blocks=4
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

function _set_algorithm(){
    if [ "$1" == "low" ];then
        algorithm=nemo_0.5_8
    else
        algorithm=nemo_0.5_16
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

while getopts ":c:i:q:r:t:d:e:o:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        e) device_data_dir="$OPTARG";;
        c) contents+=("$OPTARG");;
        i) indexes+=("$OPTARG");;
        q) qualities+=("$OPTARG");;
        r) resolutions+=("$OPTARG");;
        d) device_id="$OPTARG";;
        o) output_resolution="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${device_id+x}" ]; then
    echo "[ERROR] device_id is not set"
    exit 1;
fi

if [ -z "${device_data_dir+x}" ]; then
    echo "[ERROR] device_data_dir is not set"
    exit 1;
fi

if [ -z "${contents+x}" ]; then
    contents=("product_review" "how_to" "vlogs" "skit" "game_play" "haul" "challenge" "education" "favorite" "unboxing")
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

if [ -z "${output_resolution+x}" ]; then
    output_resolution=1080
fi

_set_conda
_set_output_size ${output_resolution}

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
                _set_algorithm ${quality}
                CUDA_VISIBLE_DEVICES=${gpu_index} python ${NEMO_ROOT}/tool/load_data_to_device.py --data_dir ${NEMO_ROOT}/data --device_data_dir ${device_data_dir} --content ${content}${index} --video_name ${resolution}p_${bitrate}kbps_s0_d300.webm --num_blocks ${num_blocks} --num_filters ${num_filters} --algorithm=${algorithm} --device_id=${device_id} --output_height=${output_height}
            done
        done
    done
done
