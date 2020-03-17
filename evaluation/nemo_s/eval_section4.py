import argparse
import os
import glob

import numpy as np

from tool.video import profile_video
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from dnn.model.nemo_s import NEMO_S
from evaluation.libvpx_results import *
from tool.mac import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str,  required=True)
    parser.add_argument('--lr_resolution', type=int, required=True)
    parser.add_argument('--hr_resolution', type=int, required=True)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    args = parser.parse_args()

    #dnn
    scale = int(args.hr_resolution // args.lr_resolution)
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)

    #cache_profiler
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random

    #log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval_section4.txt'.format(args.lr_resolution))
    with open(log_file, 'w') as f:
        cache_avg_latency = []
        cache_std_latency = []
        dnn_avg_latency = []
        dnn_std_latency = []
        video_dir = os.path.join(args.dataset_rootdir, args.content, 'video')
        lr_video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(args.lr_resolution)))[0])
        lr_video_profile = profile_video(lr_video_file)
        lr_video_name = os.path.basename(lr_video_file)

        #cache
        for i in range(75):
            log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', lr_video_name, nemo_s.name, '{}_{}'.format(aps_class.NAME1, args.threshold), 'chunk{:04d}'.format(i))
            cache_erosion = libvpx_cache_erosion(log_dir)

            f.write('{}\t{:.2f}\n'.format(i, cache_erosion[0]))
