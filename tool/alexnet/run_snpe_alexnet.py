import os, time, sys, time
import subprocess
import argparse
import collections
import json
import importlib

import numpy as np
import tensorflow as tf

from dnn.dataset import setup_images
from dnn.model.nas_s import NAS_S
from tool.snpe import snpe_convert_model, snpe_convert_dataset, snpe_benchmark, snpe_benchmark_random_config
from tool.video import profile_video, FFmpegOption

IMG_SHAPE = (224, 224, 3)
TENSOR_SHAPE = (1, 224, 224, 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--log_dir', type=str, required=True)

    #device
    parser.add_argument('--device_id', type=str)
    parser.add_argument('--runtime', type=str)

    args = parser.parse_args()

    #run benchmark
    dlc_file = os.path.join(args.log_dir, 'alexnet.dlc')
    json_file = snpe_benchmark_random_config(args.device_id, args.runtime, 'alexnet', dlc_file, args.log_dir)
    snpe_benchmark(json_file)
