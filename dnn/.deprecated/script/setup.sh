#!/bin/bash

echo $1 $2 $3

export DNNDIR=$1
export DATADIR=$2
export FFMPEGPATH=$3
export PYTHONPATH=$1:$PYTHONPATH
