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
GOP = 120

def chunk_quality(index, quality):
    start_index = index * GOP
    if GOP * (index + 1) > len(quality):
        end_index = len(quality) - 1
    else:
        end_index = (index + 1) * GOP
    return np.round(np.average(quality[start_index:end_index]), 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)

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

    #sort
    args.content.sort(key=lambda val: content_order[val])

    #cache_profiler
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random

    hr_resolution = 1080
    bilinear_resolution = [240, 360, 480, 720, 1080]
    cache_resolution = [240, 360, 480]
    num_blocks = {}
    num_filters = {}
    num_blocks[240] = 8
    num_blocks[360] = 8
    num_blocks[480] = 8
    num_filters[240] = 32
    num_filters[360] = 29
    num_filters[480] = 18

    #log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation', 'youngmok')
    os.makedirs(log_dir, exist_ok=True)
    for content in args.content:
        #bilienar
        bilinear_quality = {}
        for resolution in bilinear_resolution:
            #video, directory
            video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            if resolution == 1080:
                video_file = os.path.abspath(sorted(glob.glob(os.path.join(video_dir, '{}p*'.format(resolution))))[1])
            else:
                video_file = os.path.abspath(sorted(glob.glob(os.path.join(video_dir, '{}p*'.format(resolution))))[0])
            video_profile = profile_video(video_file)
            video_name = os.path.basename(video_file)
            bilinear_log_dir = os.path.join(args.dataset_rootdir, content, 'log', video_name)
            bilinear_quality[resolution] = libvpx_quality(bilinear_log_dir)

            #log_file= os.path.join(bilinear_log_dir, 'quality.txt')
            #if not os.path.exists(log_file):
            #    print(log_file)
            #TODO: copy

        #cache
        cache_quality = {}
        for resolution in cache_resolution:
            #video, directory
            video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(resolution)))[0])
            video_profile = profile_video(video_file)
            video_name = os.path.basename(video_file)
            scale = int(hr_resolution // resolution)
            nemo_s = NEMO_S(num_blocks[resolution], num_filters[resolution], scale, args.upsample_type)

            #latency, quality
            cache_profile_name = '{}_{}.profile'.format(aps_class.NAME1, args.threshold)
            cache_log_dir = os.path.join(args.dataset_rootdir, content, 'log', video_name, nemo_s.name, cache_profile_name)
            cache_quality[resolution] = libvpx_quality(cache_log_dir)

            #log_file = os.path.join(cache_log_dir, 'quality.txt')
            #if not os.path.exists(log_file):
            #    print(log_file)

        log_file = os.path.join(log_dir, '{}.txt'.format(content))
        with open(log_file, 'w') as f:
            for i in range(75):
                f.write('{}'.format(i))
                f.write('\t{}'.format(chunk_quality(i, cache_quality[240])))
                f.write('\t{}'.format(chunk_quality(i, cache_quality[360])))
                f.write('\t{}'.format(chunk_quality(i, cache_quality[480])))
                f.write('\t{}'.format(chunk_quality(i, bilinear_quality[240])))
                f.write('\t{}'.format(chunk_quality(i, bilinear_quality[360])))
                f.write('\t{}'.format(chunk_quality(i, bilinear_quality[480])))
                f.write('\t{}\n'.format(chunk_quality(i, bilinear_quality[720])))
                f.write('\t{}\n'.format(chunk_quality(i, bilinear_quality[1080])))
