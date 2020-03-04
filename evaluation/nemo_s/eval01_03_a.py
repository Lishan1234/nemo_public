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
from tool.mobile import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
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
    parser.add_argument('--device_name', type=str, required=True)

    args = parser.parse_args()

    #sort
    args.content.sort()

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
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation', args.device_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval01_03_a.txt')
    with open(log_file, 'w') as f:
        for content in args.content:
            lr_video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
            lr_video_profile = profile_video(lr_video_file)
            lr_video_name = os.path.basename(lr_video_file)
            fps = lr_video_profile['frame_rate']
            log_dir = os.path.join(args.dataset_rootdir, content, 'log')

            #bilienar
            bilinear_log_dir = os.path.join(log_dir, lr_video_name, args.device_name, 'monsoon')
            time, current, power = libvpx_power(os.path.join(bilinear_log_dir, 'decode.csv'))
            total_frame = libvpx_num_frames(bilinear_log_dir)
            bilinear_avg_power = np.average(power)
            bilinear_total_energy = bilinear_avg_power * time
            bilinear_avg_energy = bilinear_total_energy / total_frame
            bilinear_fps = total_frame / time
            bilinear_total_playback_time = playback_time(np.average(current), args.device_name) / (fps / bilinear_fps)

            #cache
            cache_avg_energy = []
            cache_total_playback_time = []
            for idx, threshold in enumerate(args.threshold):
                cache_profile_name = '{}_{}.profile'.format(aps_class.NAME1, threshold)
                cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name, args.device_name, 'monsoon')
                time, current, power = libvpx_power(os.path.join(cache_log_dir, 'decode_cache_{}.csv'.format(args.num_filters)))
                total_frame = libvpx_num_frames(cache_log_dir)
                cache_avg_power = np.average(power)
                cache_total_energy = cache_avg_power * time
                cache_avg_energy.append(cache_total_energy / total_frame)
                cache_fps = total_frame / time
                cache_total_playback_time.append(playback_time(np.average(current), args.device_name) / (fps / cache_fps))

            #dnn
            dnn_avg_energy = []
            dnn_total_playback_time = []
            for num_layers, num_filters in zip(args.baseline_num_blocks, args.baseline_num_filters):
                nemo_s = NEMO_S(num_layers, num_filters, scale, args.upsample_type)
                dnn_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, args.device_name, 'monsoon')
                time, current, power = libvpx_power(os.path.join(dnn_log_dir, 'decode_sr_{}.csv'.format(num_filters)))
                total_frame = libvpx_num_frames(dnn_log_dir)
                dnn_avg_power = np.average(power)
                dnn_total_energy = dnn_avg_power * time
                dnn_avg_energy.append(dnn_total_energy / total_frame)
                dnn_fps = total_frame / time
                dnn_total_playback_time.append(playback_time(np.average(current), args.device_name) / (fps / dnn_fps))

            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(content, bilinear_avg_energy,
                '\t'.join(str(x) for x in cache_avg_energy), '\t'.join(str(x) for x in dnn_avg_energy),
                bilinear_total_playback_time, '\t'.join(str(x) for x in cache_total_playback_time), '\t'.join(str(x) for x in dnn_total_playback_time)))
