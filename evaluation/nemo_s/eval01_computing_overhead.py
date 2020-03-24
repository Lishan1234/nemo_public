import argparse
import os
import glob
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

UNIT = 200
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
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--bound', type=int, default=None)
    parser.add_argument('--aps_class', type=str, required=True)

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
    elif args.aps_calss == 'nemo_bound':
        aps_class = APS_NEMO_Bound
    else:
        raise NotImplementedError

    #throughput
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval01_throughput_{}.txt'.format(id_to_name(args.device_id)))
    with open(log_file, 'w') as f0:
        for content in args.content:
            cache_avg_throughput = []
            cache_std_throughput = []
            dnn_avg_throughput = []
            dnn_std_throughput = []
            video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(args.resolution)))[0])
            video_profile = profile_video(video_file)
            video_name = os.path.basename(video_file)

            #cache
            json_file = os.path.join(args.dataset_rootdir, content, 'log', video_name, 'device_to_dnn.json')
            with open(json_file, 'r') as f1:
                json_data = json.load(f1)
            cache_nemo_s = NEMO_S(json_data[args.device_id]['num_blocks'], json_data[args.device_id]['num_filters'], json_data[args.device_id]['scale'])
            if aps_class == APS_NEMO_Bound:
                log_dir = os.path.join(args.dataset_rootdir, content, 'log', video_name, cache_nemo_s.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, args.threshold), args.device_id)
            else:
                log_dir = os.path.join(args.dataset_rootdir, content, 'log', video_name, cache_nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold), args.device_id)
            latency = libvpx_latency(log_dir)
            throughput = []
            #TODO: Calculate standard deviationi over each chunk
            for i in range(len(latency) // UNIT):
                if i == (len(latency) // UNIT - 1):
                    throughput.append(1000 / np.average(latency[i * UNIT: -1]))
                else:
                    throughput.append(1000 / np.average(latency[i * UNIT: (i + 1) * UNIT]))
            cache_avg_throughput.append(np.round(1000 / np.average(latency), 2))
            cache_std_throughput.append(np.round(np.std(throughput), 2))

            #dnn
            for num_blocks, num_filters in zip(args.baseline_num_blocks, args.baseline_num_filters):
                baseline_nemo_s = NEMO_S(num_blocks, num_filters, json_data[args.device_id]['scale'])
                log_dir = os.path.join(args.dataset_rootdir, content, 'log', video_name, baseline_nemo_s.name, args.device_id)
                latency = libvpx_latency(log_dir)
                throughput = [1000 / x for x in latency]
                dnn_avg_throughput.append(np.round(1000 / np.average(latency), 2))
                dnn_std_throughput.append(np.round(np.std(throughput), 2))

            f0.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(content, cache_nemo_s.name, '\t'.join(str(x) for x in cache_avg_throughput), \
                    '\t'.join(str(x) for x in dnn_avg_throughput), '\t'.join(str(x) for x in cache_std_throughput),
                    '\t'.join(str(x) for x in dnn_std_throughput)))

    #mac
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval01_mac.txt')
    with open(log_file, 'w') as f0:
        for content in args.content:
            cache_avg_mac = []
            dnn_avg_mac = []
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
            cache_mac, dnn_mac = libvpx_mac(log_dir)
            cache_avg_mac.append(np.round(np.average(cache_mac), 2))
            dnn_avg_mac.append(np.round(np.average(dnn_mac), 2))

            f0.write('{}\t{}\t{}\t{}\n'.format(content, cache_nemo_s.name, '\t'.join(str(x) for x in cache_avg_mac), \
                    '\t'.join(str(x) for x in dnn_avg_mac)))
