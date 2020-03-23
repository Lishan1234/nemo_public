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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--resolution', type=int, required=True)

    #dataset
    parser.add_argument('--gop', type=int, default=120)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--bound', type=int, default=None)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)

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

    #log
    latency = []
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval01_latency_{}.txt'.format(id_to_name(args.device_id)))
    with open(log_file, 'w') as f0:
        video_dir = os.path.join(args.dataset_rootdir, args.content, 'video')
        video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(args.resolution)))[0])
        video_profile = profile_video(video_file)
        video_name = os.path.basename(video_file)
        log_dir = os.path.join(args.dataset_rootdir, args.content, 'log')

        #dnn
        json_file = os.path.join(args.dataset_rootdir, args.content, 'log', video_name, 'device_to_dnn.json')
        with open(json_file, 'r') as f1:
            json_data = json.load(f1)
        nemo_s = NEMO_S(json_data[args.device_id]['num_blocks'], json_data[args.device_id]['num_filters'], json_data[args.device_id]['scale'])

        #bilienar
        bilinear_log_dir = os.path.join(log_dir, video_name)
        bilinear_latency = libvpx_latency(os.path.join(bilinear_log_dir, args.device_id))[0:120]
        latency.append(bilinear_latency)

        #cache
        if aps_class == APS_NEMO_Bound:
            cache_log_dir = os.path.join(log_dir, video_name, nemo_s.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, args.threshold))
        else:
            cache_log_dir = os.path.join(log_dir, video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold))
        cache_latency = libvpx_latency(os.path.join(cache_log_dir, args.device_id))[0:120]
        latency.append(cache_latency)

        #per-frame dnn
        dnn_latency = []
        dnn_log_dir = os.path.join(log_dir, video_name, nemo_s.name)
        dnn_latency = libvpx_latency(os.path.join(dnn_log_dir, args.device_id))[0:120]
        latency.append(dnn_latency)

        for idx, result in enumerate(zip(*latency)):
            f0.write('{}\t{}\n'.format(idx, '\t'.join(str(np.round(x, 3)) for x in result)))
