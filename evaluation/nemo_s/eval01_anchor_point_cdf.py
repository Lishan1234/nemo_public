import argparse
import os
import glob
import operator
import json

import numpy as np

from tool.video import profile_video
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from cache_profile.anchor_point_selector_nemo_bound import APS_NEMO_Bound
from dnn.model.nemo_s import NEMO_S

from evaluation.libvpx_results import *
from evaluation.cache_profile_results import *
from tool.mac import *

content_order = {'product_review': 0, 'how_to': 1, 'vlogs': 2, 'game_play': 3, 'skit': 4,
                'haul': 5, 'challenge':6, 'favorite': 7, 'education': 8, 'unboxing': 9}

def load_num_anchor_points(log_dir):
    quality_log_file = os.path.join(log_dir, 'quality.txt')
    num_anchor_points = []

    with open(quality_log_file, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            num_anchor_points.append(int(quality_line[1]))

    return num_anchor_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
    parser.add_argument('--resolution', type=int, required=True)

    #dataset
    parser.add_argument('--gop', type=int, default=120)

    #anchor point
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--bound', type=int, default=None)
    parser.add_argument('--aps_class', type=str, required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)

    args = parser.parse_args()

    #sort
    args.content.sort(key=lambda val: content_order[val])

    #log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval01_anchor_point_cdf.txt')

    #cache_profiler
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random
    elif args.aps_calss == 'nemo_bound':
        aps_class = APS_NEMO_Bound
    else:
        raise NotImplementedError

    num_anchor_points = []
    with open(log_file, 'w') as f0:
        for content in args.content:
            #video, directory
            video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(args.resolution)))[0])
            video_profile = profile_video(video_file)
            video_name = os.path.basename(video_file)

            json_file = os.path.join(args.dataset_rootdir, content, 'log', video_name, 'device_to_dnn.json')
            with open(json_file, 'r') as f1:
                json_data = json.load(f1)
            cache_nemo_s = NEMO_S(json_data[args.device_id]['num_blocks'], json_data[args.device_id]['num_filters'], json_data[args.device_id]['scale'])
            if aps_class == APS_NEMO_Bound:
                log_dir = os.path.join(args.dataset_rootdir, content, 'log', video_name, cache_nemo_s.name, '{}_{}_{}'.format(aps_class.NAME1, args.bound, args.threshold))
            else:
                log_dir = os.path.join(args.dataset_rootdir, content, 'log', video_name, cache_nemo_s.name, '{}_{}'.format(aps_class.NAME1, args.threshold))
            num_anchor_points .append(np.average(load_num_anchor_points(log_dir)))

        num_anchor_points .sort()


        print(num_anchor_points )

        count = 0
        f0.write('0\t0\n')
        for value in num_anchor_points :
            f0.write('{:.2f}\t{:.2f}\n'.format(count/len(num_anchor_points ), value))
            f0.write('{:.2f}\t{:.2f}\n'.format((count+1)/len(num_anchor_points ),value))
            count += 1
