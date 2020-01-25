import os, time, sys, time
import subprocess
import argparse
import collections
import json
import importlib

import numpy as np

from dnn.dataset import setup_images
from dnn.model.nas_s import NAS_S
from dnn.utility import FFmpegOption, resolve_bilinear
from tool.snpe import snpe_convert_model, snpe_convert_dataset, snpe_dlc_viewer, snpe_benchmark, snpe_benchmark_config, snpe_benchmark_output
from tool.ffprobe import profile_video
from tool.adb import adb_pull

def setup_dlc(model, checkpoint_dir):
    dlc_profile = snpe_convert_model(model, model.nhwc, checkpoint_dir)

    dlc_file =  os.path.join(checkpoint_dir, dlc_profile['dlc_name'])
    html_file = os.path.join(checkpoint_dir, '{}.html'.format(dlc_profile['dlc_name']))
    snpe_dlc_viewer(dlc_file, html_file)

def setup_raw_images(lr_image_dir, hr_image_dir, image_format):
    snpe_convert_dataset(lr_image_dir, image_format)
    npe_convert_dataset(hr_image_dir, image_format)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, default='/usr/bin/ffmpeg')

    #dataset
    parser.add_argument('--train_filter_type', type=str, default='uniform')
    parser.add_argument('--train_filter_fps', type=float, default=1.0)
    parser.add_argument('--test_filter_type', type=str, default='uniform')
    parser.add_argument('--test_filter_fps', type=float, default=1.0)
    parser.add_argument('--image_format', type=str, default='png')

    #architecture
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--num_blocks', type=int)

    #device
    parser.add_argument('--device_id', type=str)
    parser.add_argument('--runtime', type=str)

    #mode
    parser.add_argument('--mode', type=str, choices=['dlc', 'raw'], required=True)

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

    if args.mode == 'dlc':
        #model
        nas_s = NAS_S(args.num_blocks, args.num_filters, scale)
        model = nas_s.build_model()
        model.scale = scale
        model.nhwc = nhwc
        train_ffmpeg_option = FFmpegOption(args.train_filter_type, args.train_filter_fps, None)
        checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', train_ffmpeg_option.summary(args.lr_video_name), model.name)
        assert(os.path.exists(checkpoint_dir))

        #dlc
        setup_dlc(model, checkpoint_dir)

    elif args.mode == 'raw':
        #images
        test_ffmpeg_option = FFmpegOption(args.test_filter_type, args.test_filter_fps, None)
        lr_image_dir = os.path.join(args.dataset_dir, 'image', test_ffmpeg_option.summary(args.lr_video_name))
        hr_image_dir = os.path.join(args.dataset_dir, 'image', test_ffmpeg_option.summary(args.hr_video_name))
        setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, test_ffmpeg_option.filter())
        setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, test_ffmpeg_option.filter())

        #raw images
        setup_raw_images(lr_image_dir, hr_image_dir, args.image_format)