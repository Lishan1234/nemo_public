import argparse
import os
import glob
import operator

import numpy as np

from tool.video import profile_video
from dnn.model.nemo_s import NEMO_S

from evaluation.libvpx_results import *
from evaluation.cache_profile_results import *
from tool.mac import *
from tool.mobile import *

content_order = {'product_review': 0, 'how_to': 1, 'vlogs': 2, 'game_play': 3, 'skit': 4,
                'haul': 5, 'challenge':6, 'favorite': 7, 'education': 8, 'unboxing': 9}

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
    parser.add_argument('--baseline_num_filters', type=int, nargs='+', required=True)
    parser.add_argument('--baseline_num_blocks', type=int, nargs='+', required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)
    parser.add_argument('--device_name', type=str, required=True)

    args = parser.parse_args()

    #sort
    args.content.sort(key=lambda val: content_order[val])

    #1. latency log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation', args.device_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval_section3_01.txt'.format(args.lr_resolution))
    with open(log_file, 'w') as f:
        for content in args.content:
            #bilienar
            #video, directory
            lr_video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
            lr_video_profile = profile_video(lr_video_file)
            lr_video_name = os.path.basename(lr_video_file)
            log_dir = os.path.join(args.dataset_rootdir, content, 'log')

            #latency, quality
            bilinear_quality_log_dir = os.path.join(log_dir, lr_video_name)
            bilinear_latency_log_dir = os.path.join(bilinear_quality_log_dir, args.device_id)
            bilinear_quality = libvpx_quality(bilinear_quality_log_dir)
            bilinear_avg_latency = np.round(np.average(libvpx_latency(bilinear_latency_log_dir)), 2)
            bilinear_avg_quality = np.round(np.average(bilinear_quality), 2)

            #video, directory
            lr_video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
            lr_video_profile = profile_video(lr_video_file)
            lr_video_name = os.path.basename(lr_video_file)
            log_dir = os.path.join(args.dataset_rootdir, content, 'log')
            scale = int(args.hr_resolution // args.lr_resolution)

            #latency, quality
            dnn_avg_quality = []
            dnn_avg_latency = []
            for num_layers, num_filters in zip(args.baseline_num_blocks, args.baseline_num_filters):
                nemo_s = NEMO_S(num_layers, num_filters, scale, args.upsample_type)
                dnn_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name)
                dnn_quality = libvpx_quality(dnn_log_dir)
                dnn_avg_quality.append(np.round(np.average(dnn_quality), 3))

                dnn_latency = libvpx_latency(os.path.join(dnn_log_dir, args.device_id))
                dnn_avg_latency.append(np.round(np.average(dnn_latency), 3))

            f.write('{}\t{}\t{}\t{}\t{}\n'.format(content, bilinear_avg_quality, '\t'.join(str(x) for x in dnn_avg_quality),
                bilinear_avg_latency, '\t'.join(str(x) for x in dnn_avg_latency)))

    #2. power log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation', args.device_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval_section3_01.txt')
    with open(log_file, 'w') as f:
        for content in args.content:
            lr_video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
            lr_video_profile = profile_video(lr_video_file)
            lr_video_name = os.path.basename(lr_video_file)
            fps = lr_video_profile['frame_rate']
            log_dir = os.path.join(args.dataset_rootdir, content, 'log')

            #bilienar
            bilinear_log_dir = os.path.join('/ssd2/nemo-mobicom-backup', content, 'log', lr_video_name, args.device_name, 'monsoon')
            time, current, power = libvpx_power(os.path.join(bilinear_log_dir, 'decode.csv'))
            total_frame = libvpx_num_frames(bilinear_log_dir)
            bilinear_avg_power = np.average(power)
            bilinear_total_energy = bilinear_avg_power * time
            bilinear_avg_energy = bilinear_total_energy / total_frame
            bilinear_fps = total_frame / time
            bilinear_total_playback_time = playback_time(np.average(current), args.device_name) / (fps / bilinear_fps)

            #dnn
            dnn_avg_energy = []
            dnn_total_playback_time = []
            for num_layers, num_filters in zip(args.baseline_num_blocks, args.baseline_num_filters):
                nemo_s = NEMO_S(num_layers, num_filters, scale, args.upsample_type)
                dnn_log_dir = os.path.join('/ssd2/nemo-mobicom-backup', content, 'log', lr_video_name, nemo_s.name, args.device_name, 'monsoon')
                time, current, power = libvpx_power(os.path.join(dnn_log_dir, 'decode_sr_{}.csv'.format(num_filters)))
                total_frame = libvpx_num_frames(dnn_log_dir)
                dnn_avg_power = np.average(power)
                dnn_total_energy = dnn_avg_power * time
                dnn_avg_energy.append(dnn_total_energy / total_frame)
                dnn_fps = total_frame / time
                dnn_total_playback_time.append(playback_time(np.average(current), args.device_name) / (fps / dnn_fps))

            f.write('{}\t{}\t{}\t{}\t{}\n'.format(content, bilinear_avg_energy,
                '\t'.join(str(x) for x in dnn_avg_energy), bilinear_total_playback_time, '\t'.join(str(x) for x in dnn_total_playback_time)))

    #3. temperature log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation', args.device_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval_section3_03.txt')
    print(log_file)
    with open(log_file, 'w') as f:
        for content in args.content:
            lr_video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
            lr_video_profile = profile_video(lr_video_file)
            lr_video_name = os.path.basename(lr_video_file)
            fps = lr_video_profile['frame_rate']
            log_dir = os.path.join(args.dataset_rootdir, content, 'log')
            min_len = 0

            #bilienar
            bilinear_log_dir = os.path.join('/ssd2/nemo-mobicom-backup', content, 'log', lr_video_name, args.device_name, 'flir')
            time, temperature = libvpx_temperature(os.path.join(bilinear_log_dir, 'temperature.csv'))
            bilinear_time = time
            bilinear_temperature = temperature
            min_len = len(bilinear_time)

            #dnn
            dnn_time = []
            dnn_temperature = []
            for num_layers, num_filters in zip(args.baseline_num_blocks, args.baseline_num_filters):
                nemo_s = NEMO_S(num_layers, num_filters, scale, args.upsample_type)
                dnn_log_dir = os.path.join('/ssd2/nemo-mobicom-backup', content, 'log', lr_video_name, nemo_s.name, args.device_name, 'flir')
                time, temperature = libvpx_temperature(os.path.join(dnn_log_dir, 'temperature.csv'))
                dnn_time.append(time)
                dnn_temperature.append(temperature)
                if len(dnn_time[-1]) < min_len:
                    min_len = len(dnn_time[-1])

            for i in range(min_len):
                f.write('{:.2f}\t{:.2f}'.format(bilinear_time[i] / 1000 / 60, bilinear_temperature[i]))
                for time, temperature in zip(dnn_time, dnn_temperature):
                    f.write('\t{:.2f}\t{:.2f}'.format(time[i] / 1000 / 60, temperature[i]))
                f.write('\n')
