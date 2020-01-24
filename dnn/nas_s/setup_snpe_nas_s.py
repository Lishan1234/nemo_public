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
from tool.snpe import snpe_convert_model, snpe_convert_dataset, snpe_dlc_viewer, snpe_benchmark, snpe_download_benchmark_output
from tool.ffprobe import profile_video
from tool.adb import adb_pull
from tool.tf import valid_raw_dataset, raw_bilinear_quality, raw_sr_quality

DEVICE_ROOTDIR = '/data/local/tmp/snpebm'
BENCHMARK_CONFIG_NAME = 'benchmark.json'

def create_benchmark_config(model, log_dir, device_id, runtime, dlc_path, raw_dir, raw_list, perf='default'):
    result_dir = os.path.join(log_dir, device_id, runtime)
    json_path = os.path.join(result_dir, BENCHMARK_CONFIG_NAME)
    os.makedirs(result_dir, exist_ok=True)

    benchmark = collections.OrderedDict()
    benchmark['Name'] = model.name
    benchmark['HostRootPath'] = os.path.abspath(log_dir)
    benchmark['HostResultsDir'] = os.path.abspath(result_dir)
    benchmark['DevicePath'] = DEVICE_ROOTDIR
    benchmark['Devices'] = [device_id]
    benchmark['HostName'] = 'localhost'
    benchmark['Runs'] = 1
    benchmark['Model'] = collections.OrderedDict()
    benchmark['Model']['Name'] = model.name
    benchmark['Model']['Dlc'] = dlc_path
    benchmark['Model']['InputList'] = raw_list
    benchmark['Model']['Data'] = [raw_dir]
    benchmark['Runtimes'] = [runtime]
    benchmark['Measurements'] = ['timing']
    benchmark['ProfilingLevel'] = 'detailed'
    benchmark['BufferTypes'] = ['float']

    with open(json_path, 'w') as outfile:
        json.dump(benchmark, outfile, indent=4)

def setup_dlc(model, checkpoint_dir, log_dir):
    dlc_profile = snpe_convert_model(model, model.nhwc, checkpoint_dir)

    dlc_path =  os.path.join(checkpoint_dir, dlc_profile['dlc_name'])
    html_path = os.path.join(log_dir, '{}.html'.format(dlc_profile['dlc_name']))
    snpe_dlc_viewer(dlc_path, html_path)

def setup_raw_images(lr_image_dir, hr_image_dir, image_format):
    lr_raw_dir, lr_raw_list = snpe_convert_dataset(lr_image_dir, image_format)
    hr_raw_dir, _ = snpe_convert_dataset(hr_image_dir, image_format)

def setup_benchmark_config(model, device_id, runtime, checkpoint_dir, log_dir, lr_image_dir, hr_image_dir, image_format='png'):
    dlc_profile = snpe_convert_model(model, model.nhwc, checkpoint_dir)

    dlc_path =  os.path.join(checkpoint_dir, dlc_profile['dlc_name'])
    html_path = os.path.join(log_dir, '{}.html'.format(dlc_profile['dlc_name']))
    snpe_dlc_viewer(dlc_path, html_path)

    lr_raw_dir, lr_raw_list = snpe_convert_dataset(lr_image_dir, image_format)
    hr_raw_dir, _ = snpe_convert_dataset(hr_image_dir, image_format)

    json_path = create_benchmark_config(model, log_dir, device_id, runtime, dlc_path, lr_raw_dir, lr_raw_list)

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
    parser.add_argument('--mode', type=str, choices=['dlc', 'raw', 'benchmark'], required=True)

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
        log_dir = os.path.join(args.dataset_dir, 'log', args.lr_video_name, model.name, 'snpe')
        setup_dlc(model, checkpoint_dir, log_dir)

    elif args.mode == 'raw':
        #images
        test_ffmpeg_option = FFmpegOption(args.test_filter_type, args.test_filter_fps, None)
        lr_image_dir = os.path.join(args.dataset_dir, 'image', test_ffmpeg_option.summary(args.lr_video_name))
        hr_image_dir = os.path.join(args.dataset_dir, 'image', test_ffmpeg_option.summary(args.hr_video_name))
        setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, test_ffmpeg_option.filter())
        setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, test_ffmpeg_option.filter())

        #raw images
        setup_raw_images(lr_image_dir, hr_image_dir, args.image_format)

    elif args.mode == 'benchmark':
        #model
        nas_s = NAS_S(args.num_blocks, args.num_filters, scale)
        model = nas_s.build_model()
        model.scale = scale
        model.nhwc = nhwc
        train_ffmpeg_option = FFmpegOption(args.train_filter_type, args.train_filter_fps, None)
        checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', train_ffmpeg_option.summary(args.lr_video_name), model.name)
        assert(os.path.exists(checkpoint_dir))

        #images
        test_ffmpeg_option = FFmpegOption(args.test_filter_type, args.test_filter_fps, None)
        lr_image_dir = os.path.join(args.dataset_dir, 'image', test_ffmpeg_option.summary(args.lr_video_name))
        hr_image_dir = os.path.join(args.dataset_dir, 'image', test_ffmpeg_option.summary(args.hr_video_name))
        setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, test_ffmpeg_option.filter())
        setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, test_ffmpeg_option.filter())

        #benchmark
        log_dir = os.path.join(args.dataset_dir, 'log', args.lr_video_name, model.name, 'snpe')
        setup_benchmark_config(model, args.device_id, args.runtime, checkpoint_dir, log_dir, lr_image_dir, hr_image_dir, image_format=args.image_format)
