import argparse
import os
import glob
import operator
import sys

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

INTERVAL = 1000

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
    log_file = os.path.join(log_dir, 'eval01_04_a.txt')
    with open(log_file, 'w') as f:
        for content in args.content:
            lr_video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
            lr_video_profile = profile_video(lr_video_file)
            lr_video_name = os.path.basename(lr_video_file)
            fps = lr_video_profile['frame_rate']
            log_dir = os.path.join(args.dataset_rootdir, content, 'log')
            max_len = 0

            #bilienar
            bilinear_log_dir = os.path.join(log_dir, lr_video_name, args.device_name, 'flir')
            total_frame = libvpx_num_frames(bilinear_log_dir)
            time, temperature = libvpx_temperature(os.path.join(bilinear_log_dir, 'temperature.csv'))
            bilinear_fps = total_frame / (time[-1] - time[0]) * 1000
            time = [x * (bilinear_fps / fps) for x in time]
            bilinear_time = time
            max_len = len(bilinear_time)
            bilinear_temperature = []
            for i in range(total_frame):
                bilinear_temperature.append(temperature[int(i / total_frame * len(temperature))])
            if max_len < total_frame:
                max_len = total_frame

            #cache
            cache_times = []
            cache_temperatures = []
            for idx, threshold in enumerate(args.threshold):
                cache_profile_name = '{}_{}.profile'.format(aps_class.NAME1, threshold)
                cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name, args.device_name, 'flir')
                total_frame = libvpx_num_frames(cache_log_dir)
                time, temperature = libvpx_temperature(os.path.join(cache_log_dir, 'temperature.csv'))
                cache_fps = total_frame / (time[-1] - time[0]) * 1000
                time = [x * (cache_fps / fps) for x in time]
                cache_times.append(time)
                cache_temperature = []
                for i in range(total_frame):
                    cache_temperature.append(temperature[int(i / total_frame * len(temperature))])
                cache_temperatures.append(cache_temperature)
                if max_len < total_frame:
                    max_len = total_frame

            #dnn
            dnn_times = []
            dnn_temperatures = []
            for num_layers, num_filters in zip(args.baseline_num_blocks, args.baseline_num_filters):
                nemo_s = NEMO_S(num_layers, num_filters, scale, args.upsample_type)
                dnn_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, args.device_name, 'flir')
                total_frame = libvpx_num_frames(dnn_log_dir)
                time, temperature = libvpx_temperature(os.path.join(dnn_log_dir, 'temperature.csv'))
                dnn_fps = total_frame / (time[-1] - time[0]) * 1000
                time = [x * (dnn_fps / fps) for x in time]
                dnn_times.append(time)
                dnn_temperature = []
                for i in range(total_frame):
                    dnn_temperature.append(temperature[int(i / total_frame * len(temperature))])
                dnn_temperatures.append(dnn_temperature)
                if max_len < total_frame:
                    max_len = total_frame

            """
            for i in range(9000):
                if i < len(bilinear_time):
                    f.write('{:.2f}\t{:.2f}'.format(bilinear_time[i] / 1000 / 60, bilinear_temperature[i]))
                else:
                    f.write('\t')
                for time, temperature in zip(cache_times, cache_temperatures):
                    if i < len(time):
                        f.write('\t{:.2f}\t{:.2f}'.format(time[i] / 1000 / 60, temperature[i]))
                    else:
                        f.write('\t\t')
                for time, temperature in zip(dnn_times, dnn_temperatures):
                    if i < len(time):
                        f.write('\t{:.2f}\t{:.2f}'.format(time[i] / 1000 / 60, temperature[i]))
                    else:
                        f.write('\t\t')
                f.write('\n')
            """

            num_samples = max_len // 100
            for i in range(num_samples):
                f.write('{}'.format(i * 100))
                if i * 100 < len(bilinear_temperature):
                    f.write('\t{:.2f}'.format(bilinear_temperature[i * 100]))
                else:
                    f.write('\t')
                for temperature in cache_temperatures:
                    if i * 100 < len(temperature):
                        f.write('\t{:.2f}'.format(temperature[i * 100]))
                    else:
                        f.write('\t')
                for temperature in dnn_temperatures:
                    if i * 100 < len(temperature):
                        f.write('\t{:.2f}'.format(temperature[i * 100]))
                    else:
                        f.write('\t')
                f.write('\n')
