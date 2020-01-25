import os, time, sys, time
import subprocess
import argparse
import collections
import json
import importlib

import numpy as np
import tensorflow as tf

from dnn.dataset import setup_images
from dnn.model.nas_s import NAS_S
from dnn.utility import FFmpegOption
from tool.snpe import snpe_convert_model, snpe_convert_dataset, snpe_dlc_viewer, snpe_benchmark
from tool.ffprobe import profile_video
from tool.adb import adb_pull
from tool.tf import valid_raw_dataset, raw_bilinear_quality, raw_sr_quality

def summary(device_id, model, runtime, lr_image_dir, hr_image_dir, log_dir, perf='default'):
    #quality
    lr_raw_dir = os.path.join(lr_image_dir, 'raw')
    sr_raw_dir = os.path.join(lr_image_dir, model.name, runtime, 'raw')
    hr_raw_dir = os.path.join(hr_image_dir, 'raw')
    bilinear_psnr= raw_bilinear_quality(lr_raw_dir, hr_raw_dir, model.nhwc, model.scale)
    sr_psnr= raw_sr_quality(sr_raw_dir, hr_raw_dir, model.nhwc, model.scale)
    avg_bilinear_psnr = np.average(bilinear_psnr)
    avg_sr_psnr = np.average(sr_psnr)

    #latency
    benchmark_log_dir = os.path.join(log_dir, model.name, 'snpe', device_id, runtime)
    result_json_file = os.path.join(benchmark_log_dir, 'latest_results', 'benchmark_stats_{}.json'.format(model.name))
    assert(os.path.exists(result_json_file))
    with open(result_json_file, 'r') as f:
        json_data = json.load(f)
        avg_latency = float(json_data['Execution_Data']['GPU_FP16']['Total Inference Time']['Avg_Time']) / 1000

    #size
    config_json_file = os.path.join(benchmark_log_dir, 'benchmark.json')
    with open(config_json_file, 'r') as f:
        json_data = json.load(f)
        dlc_file = json_data['Model']['Dlc']
        size = os.path.getsize(dlc_file) / 1000

    #log
    summary_log_file = os.path.join(benchmark_log_dir, 'summary.txt')
    with open(summary_log_file, 'w') as f:
        f.write('PSNR (dB)\t{:.2f}\t{:.2f}\n'.format(avg_sr_psnr, avg_bilinear_psnr))
        f.write('Latency (msec)\t{:.2f}\n'.format(avg_latency))
        f.write('Size (KB)\t{:.2f}\n'.format(size))

    return avg_sr_psnr, avg_bilinear_psnr, avg_latency, size

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, default='/usr/bin/ffmpeg')

    #dataset
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)
    parser.add_argument('--image_format', type=str, default='png')

    #architecture
    parser.add_argument('--num_filters', type=int, nargs='+')
    parser.add_argument('--num_blocks', type=int, nargs='+')

    #device
    parser.add_argument('--device_id', type=str)
    parser.add_argument('--runtime', type=str)

    args = parser.parse_args()

    #scale & nhwc
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))
    lr_video_profile = profile_video(lr_video_path)
    hr_video_profile = profile_video(hr_video_path)
    scale = hr_video_profile['height'] // lr_video_profile['height']
    nhwc = [1, lr_video_profile['height'], lr_video_profile['width'], 3]

    #log directory
    log_dir = os.path.join(args.dataset_dir, 'log', args.lr_video_name)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'summary_snpe_{}_{}.txt'.format(args.device_id, args.runtime))
    with open(log_file, 'w') as f:
        f.write('Model Name\tSR PSNR (dB)\tBilinear PSNR (dB)\tLatency (msec)\tSize (KB)\n')
        for num_blocks, num_filters in zip(args.num_blocks, args.num_filters):
            nas_s = NAS_S(num_blocks, num_filters, scale)
            model = nas_s.build_model()
            model.scale = scale
            model.nhwc = nhwc

            ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
            lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name))
            hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.hr_video_name))
            sr_psnr, bilinear_psnr, latency, size = summary(args.device_id, model, args.runtime, lr_image_dir, hr_image_dir, log_dir, perf='default')
            f.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(model.name, sr_psnr, bilinear_psnr, latency, size))
