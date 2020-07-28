#!/bin/bash

function _set_conda(){
    source ~/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate nemo_py3.6
}

_set_conda
python ${NEMO_ROOT}/cache_profile/analyze.py --data_dir ${NEMO_ROOT}/data
