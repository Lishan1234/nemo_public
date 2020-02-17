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
from tool.snpe import snpe_benchmark_result
from tool.video import FFmpegOption, profile_video

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
    parser.add_argument('--upsample_type', type=str, required=True)

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

    #ffmpeg option
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)

    #summary
    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.hr_video_name))
    log_file = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(args.lr_video_name), 'summary_snpe_{}_{}.txt'.format(args.device_id, args.runtime))
    with open(log_file, 'w') as f:
        f.write('Model Name\tSR PSNR (dB)\tBilinear PSNR (dB)\tLatency (msec)\tSize (KB)\n')
        for num_blocks, num_filters in zip(args.num_blocks, args.num_filters):
            nas_s = NAS_S(num_blocks, num_filters, scale, args.upsample_type)
            model = nas_s.build_model()
            model.scale = scale
            model.nhwc = nhwc

            log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(args.lr_video_name), model.name, 'snpe', args.device_id, args.runtime)
            os.makedirs(log_dir, exist_ok=True)
            sr_psnr, bilinear_psnr, latency, size = snpe_benchmark_result(args.device_id, args.runtime, model, lr_image_dir, hr_image_dir, log_dir, perf='default')
            f.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(model.name, sr_psnr, bilinear_psnr, latency, size))
            f.flush()

            print('Summary: Finish {}'.format(model.name))
