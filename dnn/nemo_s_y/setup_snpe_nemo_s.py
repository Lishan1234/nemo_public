import os, time, sys, time
import subprocess
import argparse
import collections
import json
import importlib

import numpy as np

from dnn.dataset import setup_yuv_images
from dnn.model.nemo_s_y import NEMO_S_Y
from tool.snpe import snpe_convert_model, snpe_dlc_viewer, snpe_convert_dataset
from tool.video import profile_video, FFmpegOption

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--vpxdec_file', type=str, required=True)

    #dataset
    parser.add_argument('--train_filter_type', type=str, default='uniform')
    parser.add_argument('--train_filter_fps', type=float, default=1.0)
    parser.add_argument('--test_filter_type', type=str, default='uniform')
    parser.add_argument('--test_filter_fps', type=float, default=1.0)
    parser.add_argument('--image_format', type=str, default='png')

    #architecture
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--upsample_type', type=str, required=True)

    #device
    parser.add_argument('--device_id', type=str)
    parser.add_argument('--runtime', type=str)

    #mode
    parser.add_argument('--mode', type=str, choices=['model', 'image', 'all'], required=True)

    args = parser.parse_args()

    #scale & nhwc
    lr_video_file = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_file = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_file))
    assert(os.path.exists(hr_video_file))
    lr_video_profile = profile_video(lr_video_file)
    hr_video_profile = profile_video(hr_video_file)
    scale = hr_video_profile['height'] // lr_video_profile['height']
    nhwc = [1, lr_video_profile['height'], lr_video_profile['width'], 1]

    if args.mode == 'model':
        #model
        nemo_s_y = NEMO_S_Y(args.num_blocks, args.num_filters, scale, args.upsample_type)
        if (hr_video_profile['height'] % lr_video_profile['height'] == 0 and
                hr_video_profile['width'] % lr_video_profile['width'] == 0):
            model = nemo_s_y.build_model()
        else:
            model = nemo_s_y.build_model(resolution=(hr_video_profile['height'], hr_video_profile['width']))
        model = nemo_s_y.build_model()
        model.scale = scale
        model.nhwc = nhwc
        train_ffmpeg_option = FFmpegOption(args.train_filter_type, args.train_filter_fps, None)
        checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', train_ffmpeg_option.summary(args.lr_video_name), model.name)
        assert(os.path.exists(checkpoint_dir))

        #dlc
        snpe_convert_model(model, model.nhwc, checkpoint_dir)

    elif args.mode == 'image':
        #images
        test_ffmpeg_option = FFmpegOption(args.test_filter_type, args.test_filter_fps, None)
        setup_yuv_images(args.vpxdec_file, args.dataset_dir, lr_video_file, args.test_filter_fps)
        setup_yuv_images(args.vpxdec_file, args.dataset_dir, hr_video_file, args.test_filter_fps)
        lr_image_dir = os.path.join(args.dataset_dir, 'image', args.lr_video_name, 'libvpx')
        hr_image_dir = os.path.join(args.dataset_dir, 'image', args.hr_video_name, 'libvpx')

        #raw images
        snpe_convert_dataset(lr_image_dir, args.image_format)
        snpe_convert_dataset(hr_image_dir, args.image_format)

    elif args.mode == 'all':
        #model
        nemo_s_y = NEMO_S_Y(args.num_blocks, args.num_filters, scale, args.upsample_type)
        if (hr_video_profile['height'] % lr_video_profile['height'] == 0 and
                hr_video_profile['width'] % lr_video_profile['width'] == 0):
            model = nemo_s_y.build_model()
        else:
            model = nemo_s_y.build_model(resolution=(hr_video_profile['height'], hr_video_profile['width']))
        model.scale = scale
        model.nhwc = nhwc
        train_ffmpeg_option = FFmpegOption(args.train_filter_type, args.train_filter_fps, None)
        checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', train_ffmpeg_option.summary(args.lr_video_name), model.name)
        assert(os.path.exists(checkpoint_dir))

        #dlc
        snpe_convert_model(model, model.nhwc, checkpoint_dir)

        #images
        test_ffmpeg_option = FFmpegOption(args.test_filter_type, args.test_filter_fps, None)
        setup_yuv_images(args.vpxdec_file, args.dataset_dir, lr_video_file, args.test_filter_fps)
        setup_yuv_images(args.vpxdec_file, args.dataset_dir, hr_video_file, args.test_filter_fps)
        lr_image_dir = os.path.join(args.dataset_dir, 'image', args.lr_video_name, 'libvpx')
        hr_image_dir = os.path.join(args.dataset_dir, 'image', args.hr_video_name, 'libvpx')

        #raw images
        snpe_convert_dataset(lr_image_dir, args.image_format)
        snpe_convert_dataset(hr_image_dir, args.image_format)
