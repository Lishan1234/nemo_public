import argparse
import os
import glob
import operator

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
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr_resolution', type=int, required=True)
    parser.add_argument('--hr_resolution', type=int, required=True)

    #dataset
    parser.add_argument('--gop', type=int, default=120)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, nargs='+', required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)
    parser.add_argument('--bound', type=int, default=None)

    #device
    parser.add_argument('--device_id', type=str, required=True)

    args = parser.parse_args()

    #dnn
    scale = int(args.hr_resolution // args.lr_resolution)
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)

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
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval01_tradeoff_{}.txt'.format(id_to_name(args.device_id)))
    with open(log_file, 'w') as f:
        lr_video_dir = os.path.join(args.dataset_rootdir, args.content, 'video')
        lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
        lr_video_profile = profile_video(lr_video_file)
        lr_video_name = os.path.basename(lr_video_file)
        fps = lr_video_profile['frame_rate']
        log_dir = os.path.join(args.dataset_rootdir, args.content, 'log')

        #bilienar (power)
        no_dnn_log_dir = os.path.join(log_dir, lr_video_name, id_to_name(args.device_id))
        time, current, power = libvpx_power(os.path.join(no_dnn_log_dir, 'monsoon', 'decode.csv'))
        no_dnn_power = np.average(power) / 1000

        no_dnn_quality = None
        nemo_power = []
        nemo_quality = []
        for idx, threshold in enumerate(args.threshold):
            #cache (power)
            if aps_class == APS_NEMO_Bound:
                log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', lr_video_name, nemo_s.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, threshold), id_to_name(args.device_id))
            else:
                log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', lr_video_name, nemo_s.name, '{}_{}.profile'.format(aps_class.NAME1, threshold), id_to_name(args.device_id))
            time, current, power = libvpx_power(os.path.join(log_dir, 'monsoon', 'decode_cache_{}.csv'.format(args.num_filters)))
            nemo_power.append(np.average(power) / 1000)

            #cache (power)
            if aps_class == APS_NEMO_Bound:
                log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', lr_video_name, nemo_s.name, '{}_{}_{}'.format(aps_class.NAME1, args.bound, threshold))
            else:
                log_dir = os.path.join(args.dataset_rootdir, args.content, 'log', lr_video_name, nemo_s.name, '{}_{}'.format(aps_class.NAME1, threshold))
            cache_quality, _, bilinear_quality = load_quality(log_dir)
            nemo_quality.append(np.average(cache_quality))

            if no_dnn_quality is None:
                no_dnn_quality = np.average(bilinear_quality)

        f.write('No DNN\t{}\t{}\n'.format(no_dnn_power, no_dnn_quality))
        for power, quality in zip(nemo_power, nemo_quality):
            f.write('NEMO\t{}\t{}\n'.format(power, quality))
