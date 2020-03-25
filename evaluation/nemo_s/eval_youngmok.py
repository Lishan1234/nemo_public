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
from tool.mobile import *

content_order = {'product_review': 0, 'how_to': 1, 'vlogs': 2, 'game_play_1': 3, 'skit': 4,
                'haul': 5, 'challenge':6, 'favorite': 7, 'education': 8, 'unboxing': 9}
GOP = 120

def load_quality(log_dir):
    quality_log_file = os.path.join(log_dir, 'quality.txt')
    quality_cache = []
    quality_dnn = []
    quality_bilinear = []

    with open(quality_log_file, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality_cache.append(float(quality_line[2]))
            quality_dnn.append(float(quality_line[3]))
            quality_bilinear.append(float(quality_line[4]))

    return quality_cache, quality_dnn, quality_bilinear

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

    #validation
    if args.aps_class == 'nemo_bound':
        assert(args.bound is not None)

    #sort
    args.content.sort(key=lambda val: content_order[val])

    #hard-coded configuration
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
    for content in args.content:
        #bilienar
        bilinear_quality = {}
        for resolution in bilinear_resolution:
            video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            if resolution == 1080:
                video_file = os.path.abspath(sorted(glob.glob(os.path.join(video_dir, '{}p*'.format(resolution))))[1])
            else:
                video_file = os.path.abspath(sorted(glob.glob(os.path.join(video_dir, '{}p*'.format(resolution))))[0])
            video_profile = profile_video(video_file)
            video_name = os.path.basename(video_file)
            bilinear_log_dir = os.path.join(args.dataset_rootdir, content, 'log', video_name)
            bilinear_quality[resolution] = libvpx_quality(bilinear_log_dir)

        #cache
        cache_quality = {}
        for resolution in cache_resolution:
            video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(resolution)))[0])
            video_profile = profile_video(video_file)
            video_name = os.path.basename(video_file)
            scale = int(hr_resolution // resolution)
            json_file = os.path.join(args.dataset_rootdir, content, 'log', video_name, 'device_to_dnn.json')
            with open(json_file, 'r') as f1:
                json_data = json.load(f1)
            nemo_s = NEMO_S(json_data[args.device_id]['num_blocks'], json_data[args.device_id]['num_filters'], json_data[args.device_id]['scale'])

            if json_data[args.device_id]['aps_class'] == 'nemo':
                aps_class = APS_NEMO
            elif json_data[args.device_id]['aps_class'] == 'uniform':
                aps_class = APS_Uniform
            elif json_data[args.device_id]['aps_class'] == 'random':
                aps_class = APS_Random
            elif json_data[args.device_id]['aps_class'] == 'nemo_bound':
                aps_class = APS_NEMO_Bound
            else:
                raise NotImplementedError

            if aps_class == APS_NEMO_Bound:
                cache_profile_name = '{}_{}_{}.profile'.format(aps_class.NAME1, json_data[args.device_id]['bound'], json_data[args.device_id]['threshold'])
            else:
                cache_profile_name = '{}_{}.profile'.format(aps_class.NAME1, json_data[args.device_id]['threshold'])

            if aps_class == APS_NEMO_Bound:
                cache_log_dir = os.path.join(args.dataset_rootdir, content, 'log', video_name, nemo_s.name, '{}_{}_{}'.format(aps_class.NAME1, json_data[args.device_id]['bound'], json_data[args.device_id]['threshold']))
            else:
                cache_log_dir = os.path.join(args.dataset_rootdir, content, 'log', video_name, nemo_s.name, '{}_{}'.format(aps_class.NAME1, json_data[args.device_id]['threshold']))
            cache_quality[resolution], _, _ = load_quality(cache_log_dir)

        log_dir = os.path.join(args.dataset_rootdir, 'evaluation', 'youngmok', id_to_name(args.device_id))
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, '{}.txt'.format(content))
        with open(log_file, 'w') as f:
            for i in range(75):
                f.write('{}'.format(i))
                f.write('\t{}'.format(cache_quality[240][i]))
                f.write('\t{}'.format(cache_quality[360][i]))
                f.write('\t{}'.format(cache_quality[480][i]))
                f.write('\t{}'.format(chunk_quality(i, bilinear_quality[240])))
                f.write('\t{}'.format(chunk_quality(i, bilinear_quality[360])))
                f.write('\t{}'.format(chunk_quality(i, bilinear_quality[480])))
                f.write('\t{}\n'.format(chunk_quality(i, bilinear_quality[720])))
                f.write('\t{}\n'.format(chunk_quality(i, bilinear_quality[1080])))
