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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr_resolution', type=int, required=True)
    parser.add_argument('--hr_resolution', type=int, required=True)

    #dataset
    parser.add_argument('--gop', type=int, default=120)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--baseline_num_filters', type=int, nargs='+', required=True)
    parser.add_argument('--baseline_num_blocks', type=int, nargs='+', required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, nargs='+', required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)

    args = parser.parse_args()

    #validation
    assert(args.num_filters == args.baseline_num_filters[-1])
    assert(args.num_blocks == args.baseline_num_blocks[-1])

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
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation', args.device_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval01_02_b_{}p.txt'.format(args.lr_resolution))
    with open(log_file, 'w') as f:
        lr_video_dir = os.path.join(args.dataset_rootdir, args.content, 'video')
        lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
        lr_video_profile = profile_video(lr_video_file)
        lr_video_name = os.path.basename(lr_video_file)
        log_dir = os.path.join(args.dataset_rootdir, args.content, 'log')

        latency = []

        #bilienar
        bilinear_log_dir = os.path.join(log_dir, lr_video_name)
        bilinear_latency = libvpx_latency(os.path.join(bilinear_log_dir, args.device_id))[1000:1120]
        latency.append(bilinear_latency)

        #cache
        cache_latency = []
        for idx, threshold in enumerate(args.threshold):
            cache_profile_name = '{}_{}.profile'.format(aps_class.NAME1, threshold)
            cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, threshold))
            cache_latency = libvpx_latency(os.path.join(cache_log_dir, args.device_id))[1000:1120]
            latency.append(cache_latency)

        #dnn
        dnn_latency = []
        for num_layers, num_filters in zip(args.baseline_num_blocks, args.baseline_num_filters):
            nemo_s = NEMO_S(num_layers, num_filters, scale, args.upsample_type)
            dnn_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name)
            dnn_latency = libvpx_latency(os.path.join(dnn_log_dir, args.device_id))[1000:1120]
            latency.append(dnn_latency)

        for result in zip(*latency):
            f.write('{}\t{}\n'.format(args.content, '\t'.join(str(np.round(x, 3)) for x in result)))
