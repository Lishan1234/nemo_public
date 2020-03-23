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
from tool.mac import *
from tool.mobile import *

content_order = {'product_review': 0, 'how_to': 1, 'vlogs': 2, 'game_play': 3, 'skit': 4,
                'haul': 5, 'challenge':6, 'favorite': 7, 'education': 8, 'unboxing': 9}

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
    parser.add_argument('--resolution', type=int, required=True)

    #dnn
    parser.add_argument('--baseline_num_filters', type=int, nargs='+', required=True)
    parser.add_argument('--baseline_num_blocks', type=int, nargs='+', required=True)

    #anchor point selector
    parser.add_argument('--aps_class', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--bound', type=int, default=None)

    #device
    parser.add_argument('--device_id', type=str, required=True)

    args = parser.parse_args()

    #validation
    if args.aps_class == 'nemo_bound':
        assert(args.bound is not None)

    #sort
    args.content.sort(key=lambda val: content_order[val])

    #cache_profiler
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random
    elif args.aps_class == 'nemo_bound':
        aps_class = APS_NEMO_Bound
    else:
        raise NotImplementedError

    #log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file0 = os.path.join(log_dir, 'eval01_quality_gain_{}.txt'.format(id_to_name(args.device_id)))
    log_file1 = os.path.join(log_dir, 'eval01_quality_{}.txt'.format(id_to_name(args.device_id)))
    with open(log_file0, 'w') as f0, open(log_file1, 'w') as f1:
        for content in args.content:
            video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(args.resolution)))[0])
            video_profile = profile_video(video_file)
            video_name = os.path.basename(video_file)
            log_dir = os.path.join(args.dataset_rootdir, content, 'log')

            #bilinear
            bilinear_avg_quality = []
            bilinear_std_quality = []

            #cache
            cache_avg_quality_gain = []
            cache_std_quality_gain = []
            cache_avg_quality = []
            cache_std_quality = []

            json_file = os.path.join(args.dataset_rootdir, content, 'log', video_name, 'device_to_dnn.json')
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            cache_nemo_s = NEMO_S(json_data[args.device_id]['num_blocks'], json_data[args.device_id]['num_filters'], json_data[args.device_id]['scale'])

            if args.aps_class == 'nemo_bound':
                cache_profile_name = '{}_{}_{}'.format(aps_class.NAME1, args.bound, args.threshold)
            else:
                cache_profile_name = '{}_{}'.format(aps_class.NAME1, args.threshold)
            cache_log_dir = os.path.join(log_dir, video_name, cache_nemo_s.name, cache_profile_name)

            cache_quality, _, bilinear_quality = load_quality(cache_log_dir)
            cache_quality_gain = list(map(operator.sub, cache_quality, bilinear_quality))
            cache_avg_quality_gain.append(np.round(np.average(cache_quality_gain), 2))
            cache_std_quality_gain.append(np.round(np.std(cache_quality_gain), 2))
            cache_avg_quality.append(np.round(np.average(cache_quality), 2))
            cache_std_quality.append(np.round(np.std(cache_quality), 2))
            bilinear_avg_quality.append(np.round(np.average(bilinear_quality), 2))
            bilinear_std_quality.append(np.round(np.std(bilinear_quality), 2))

            #dnn
            dnn_avg_quality_gain = []
            dnn_std_quality_gain = []
            dnn_avg_quality = []
            dnn_std_quality = []
            for num_blocks, num_filters in zip(args.baseline_num_blocks, args.baseline_num_filters):
                nemo_s = NEMO_S(num_blocks, num_filters, json_data[args.device_id]['scale'])
                if args.aps_class == 'nemo_bound':
                    cache_profile_name = '{}_{}_{}'.format(aps_class.NAME1, args.bound, args.threshold)
                else:
                    cache_profile_name = '{}_{}'.format(aps_class.NAME1, args.threshold)
                cache_log_dir = os.path.join(log_dir, video_name, nemo_s.name, cache_profile_name)
                _, dnn_quality , bilinear_quality = load_quality(cache_log_dir)
                dnn_quality_gain = list(map(operator.sub, dnn_quality, bilinear_quality))
                dnn_avg_quality_gain.append(np.round(np.average(dnn_quality_gain), 2))
                dnn_std_quality_gain.append(np.round(np.std(dnn_quality_gain), 2))
                dnn_avg_quality.append(np.round(np.average(dnn_quality), 2))
                dnn_std_quality.append(np.round(np.std(dnn_quality), 2))

            #log: quality gain
            f0.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(content, cache_nemo_s.name, '\t'.join(str(x) for x in cache_avg_quality_gain),
                '\t'.join(str(x) for x in dnn_avg_quality_gain), '\t'.join(str(x) for x in cache_std_quality_gain),
                '\t'.join(str(x) for x in dnn_std_quality_gain)))

            #log: absolute quality
            f1.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(content, cache_nemo_s.name, '\t'.join(str(x) for x in cache_avg_quality),
                '\t'.join(str(x) for x in dnn_avg_quality), '\t'.join(str(x) for x in cache_std_quality),
                '\t'.join(str(x) for x in dnn_std_quality)))
