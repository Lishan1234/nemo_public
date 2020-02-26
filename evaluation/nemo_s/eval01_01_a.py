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

content_order = {'product_review': 0, 'how_to': 1, 'vlogs': 2, 'game_play': 3, 'skit': 4,
                'haul': 5, 'challenge':6, 'favorite': 7, 'education': 8, 'unboxing': 9}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
    parser.add_argument('--lr_resolution', type=int, required=True)
    parser.add_argument('--hr_resolution', type=int, required=True)

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

    #sort
    args.content.sort(key=lambda val: content_order[val])

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
    log_file = os.path.join(log_dir, 'eval01_01_a_{}p.txt'.format(args.lr_resolution))
    with open(log_file, 'w') as f:
        for content in args.content:
            cache_avg_latency = []
            cache_std_latency = []
            dnn_avg_latency = []
            dnn_std_latency = []
            video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            lr_video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(args.lr_resolution)))[0])
            lr_video_profile = profile_video(lr_video_file)
            lr_video_name = os.path.basename(lr_video_file)

            #cache
            for idx, threshold in enumerate(args.threshold):
                log_dir = os.path.join(args.dataset_rootdir, content, 'log', lr_video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, threshold), args.device_id)
                latency = libvpx_latency(log_dir)
                cache_avg_latency.append(np.round(np.average(latency), 2))
                cache_std_latency.append(np.round(np.std(latency), 2))

            for num_layers, num_filters in zip(args.baseline_num_blocks, args.baseline_num_filters):
                nemo_s = NEMO_S(num_layers, num_filters, scale, args.upsample_type)
                log_dir = os.path.join(args.dataset_rootdir, content, 'log', lr_video_name, nemo_s.name, args.device_id)
                latency = libvpx_latency(log_dir)
                dnn_avg_latency.append(np.round(np.average(latency), 2))
                dnn_std_latency.append(np.round(np.std(latency), 2))

            f.write('{}\t{}\t{}\t{}\t{}\n'.format(content, '\t'.join(str(x) for x in cache_avg_latency), \
                    '\t'.join(str(x) for x in dnn_avg_latency), '\t'.join(str(x) for x in cache_std_latency),
                    '\t'.join(str(x) for x in dnn_std_latency)))
