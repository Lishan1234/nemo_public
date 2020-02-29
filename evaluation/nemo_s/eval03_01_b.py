import argparse
import os
import glob
import operator

import numpy as np

from tool.video import profile_video
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from dnn.model.nemo_s import NEMO_S

from evaluation.libvpx_results import *
from evaluation.cache_profile_results import *
from tool.mac import *

content_order = {'product_review': 0, 'how_to': 1, 'vlogs': 2, 'game_play': 3, 'skit': 4,
                'haul': 5, 'challenge':6, 'favorite': 7, 'education': 8, 'unboxing': 9}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)

    #dataset
    parser.add_argument('--gop', type=int, default=120)

    #dnn
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)

    args = parser.parse_args()

    #setting
    lr_resolutions = [240, 360, 480]
    hr_resolution = 1080
    num_blocks = {}
    num_blocks[240] = 8
    num_blocks[360] = 8
    num_blocks[480] = 8
    num_filters = {}
    num_filters[240] = 32
    num_filters[360] = 29
    num_filters[480] = 18

    #cache_profiler
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random

    #log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation', args.device_id)
    log_file = os.path.join(log_dir, 'eval03_01_b.txt')
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, 'w') as f:
        for lr_resolution in lr_resolutions:
            #video, directory
            lr_video_dir = os.path.join(args.dataset_rootdir, args.content, 'video')
            lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(lr_resolution)))[0])
            lr_video_name = os.path.basename(lr_video_file)
            log_dir = os.path.join(args.dataset_rootdir, args.content, 'log')
            scale = int(hr_resolution // lr_resolution)
            nemo_s = NEMO_S(num_blocks[lr_resolution], num_filters[lr_resolution], scale, args.upsample_type)

            #latency
            cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold))
            decode, bilinear_interpolation, motion_compensation= libvpx_breakdown_latency(os.path.join(cache_log_dir, args.device_id))
            f.write('{}p\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(lr_resolution, np.average(decode), np.average(bilinear_interpolation), np.average(motion_compensation)))
