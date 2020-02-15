import time
import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from dnn.dataset import valid_image_dataset, single_image_dataset, setup_images
from dnn.model.nas_s import NAS_S
from dnn.test import SingleTester
from dnn.utility import resolve, resolve_bilinear
from tool.video import profile_video, VideoMetadata, FFmpegOption

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

    #video metadata
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)

    #dataset
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--load_on_memory', action='store_true')

    #architecture
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)

    #log
    parser.add_argument('--custom_tag', type=str, default=None)
    parser.add_argument('--save_image', action='store_true')

    args = parser.parse_args()

    #scale
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))
    lr_video_profile = profile_video(lr_video_path)
    hr_video_profile = profile_video(hr_video_path)
    scale = hr_video_profile['height'] // lr_video_profile['height']

    #image
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #dnn
    nas_s = NAS_S(args.num_blocks, args.num_filters, scale)
    if (hr_video_profile['height'] % lr_video_profile['height'] == 0 and
            hr_video_profile['width'] % lr_video_profile['width'] == 0):
        model = nas_s.build_model()
    else:
        model = nas_s.build_model(resolution=(hr_video_profile['height'], hr_video_profile['width']))

    #dataset
    valid_ds = valid_image_dataset(lr_image_dir, hr_image_dir)

    #trainer
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), model.name)
    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(args.lr_video_name), model.name)
    image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name), model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    tester = SingleTester(model, checkpoint_dir, log_dir, image_dir)
    tester.test(valid_ds, args.save_image)
