#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS] [-i INDEXES]

optional multiple arguments:
-c CONTENTS                 Specifies contents (e.g., product_review0)
-i INDEXES                  Specifies indexes (e.g., 1)

EOF
}

function _set_conda(){
    source ~/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate nemo_py3.6
}

while getopts ":c:i:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        i) indexes+=("$OPTARG");;
        c) contents+=("$OPTARG");;
        \?) exit 1;
    esac
done

if [ -z "${indexes+x}" ]; then
    indexes=("1" "2" "3")
fi

if [ -z "${contents+x}" ]; then
    contents=("product_review" "how_to" "vlogs" "skit" "game_play" "haul" "challenge" "education" "favorite" "unboxing")
fi

for content in "${contents[@]}"
do
    for index in "${indexes[@]}"
    do
        python ${NEMO_ROOT}/tool/downloader.py --video_dir ${NEMO_ROOT}/data/video --content ${content} --index ${index}
    done
done
