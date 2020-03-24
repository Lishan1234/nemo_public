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

UNIT = 400

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_backup_rootdir', type=str, required=True)
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--resolution', type=int, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--bound', type=int, default=None)
    parser.add_argument('--aps_class', type=str, required=True)

    args = parser.parse_args()

    #validation
    if args.aps_class == 'nemo_bound':
        assert(args.bound is not None)

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
    log_dir = os.path.join(args.dataset_backup_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval01_multiple_devices.txt')
    with open(log_file, 'w') as f:
        video_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'video')
        video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(args.resolution)))[0])
        video_profile = profile_video(video_file)
        video_name = os.path.basename(video_file)

        #s10+
        device_name = 's10+'
        nemo_s = NEMO_S(8, 32, 4)
        if aps_class == APS_NEMO_Bound:
            log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, args.threshold), device_name)
        else:
            log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold), device_name)
        latency = libvpx_latency(log_dir)
        cache_avg_throughput = np.round(1000 / np.average(latency), 2)
        log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, device_name)
        latency = libvpx_latency(log_dir)
        dnn_avg_throughput = np.round(1000 / np.average(latency), 2)
        f.write('s10+\t{}\t{:.2f}\t{:.2f}\n'.format(nemo_s.name, cache_avg_throughput, dnn_avg_throughput))

        #note8
        device_name = 'note8'
        nemo_s = NEMO_S(8, 21, 4)
        if aps_class == APS_NEMO_Bound:
            log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, args.threshold), device_name)
        else:
            log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold), device_name)
        latency = libvpx_latency(log_dir)
        cache_avg_throughput = np.round(1000 / np.average(latency), 2)
        log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, device_name)
        latency = libvpx_latency(log_dir)
        dnn_avg_throughput = np.round(1000 / np.average(latency), 2)
        f.write('note8\t{}\t{:.2f}\t{:.2f}\n'.format(nemo_s.name, cache_avg_throughput, dnn_avg_throughput))

        #s9
        device_name = 's9'
        nemo_s = NEMO_S(8, 21, 4)
        if aps_class == APS_NEMO_Bound:
            log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, args.threshold), device_name)
        else:
            log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold), device_name)
        latency = libvpx_latency(log_dir)
        cache_avg_throughput = np.round(1000 / np.average(latency), 2)
        log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, device_name)
        latency = libvpx_latency(log_dir)
        dnn_avg_throughput = np.round(1000 / np.average(latency), 2)
        f.write('s9\t{}\t{:.2f}\t{:.2f}\n'.format(nemo_s.name, cache_avg_throughput, dnn_avg_throughput))

        #gpad
        device_name = 'LMT605728961d9'
        nemo_s = NEMO_S(8, 21, 4)
        if aps_class == APS_NEMO_Bound:
            log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, args.threshold), device_name)
        else:
            log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold), device_name)
        latency = libvpx_latency(log_dir)
        cache_avg_throughput = np.round(1000 / np.average(latency), 2)
        log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', video_name, nemo_s.name, device_name)
        latency = libvpx_latency(log_dir)
        dnn_avg_throughput = np.round(1000 / np.average(latency), 2)
        f.write('gpad\t{}\t{:.2f}\t{:.2f}\n'.format(nemo_s.name, cache_avg_throughput, dnn_avg_throughput))

        #a70
        device_name = 'a70'
        nemo_s = NEMO_S(8, 9, 4)
        if aps_class == APS_NEMO_Bound:
            log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, args.threshold), device_name)
        else:
            log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold), device_name)
        latency = libvpx_latency(log_dir)
        cache_avg_throughput = np.round(1000 / np.average(latency), 2)
        log_dir = os.path.join(args.dataset_backup_rootdir, args.content, 'log', video_name, nemo_s.name, device_name)
        latency = libvpx_latency(log_dir)
        dnn_avg_throughput = np.round(1000 / np.average(latency), 2)
        f.write('s10+\t{}\t{:.2f}\t{:.2f}\n'.format(nemo_s.name, cache_avg_throughput, dnn_avg_throughput))

        #redmi
        device_name = '10098e40'
        nemo_s = NEMO_S(8, 9, 4)
        if aps_class == APS_NEMO_Bound:
            log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, args.threshold), device_name)
        else:
            log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold), device_name)
        latency = libvpx_latency(log_dir)
        cache_avg_throughput = np.round(1000 / np.average(latency), 2)
        log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', video_name, nemo_s.name, device_name)
        latency = libvpx_latency(log_dir)
        dnn_avg_throughput = np.round(1000 / np.average(latency), 2)
        f.write('redmi\t{}\t{:.2f}\t{:.2f}\n'.format(nemo_s.name, cache_avg_throughput, dnn_avg_throughput))
