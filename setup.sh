#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

echo "source $SCRIPTPATH/envsetup.sh -t $SCRIPTPATH/third_party/tensorflow" >> $HOME/.bashrc
