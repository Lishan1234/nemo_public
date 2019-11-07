#Training script for a single resolution network

import argparse
import os
import sys
from importlib import import_module

from model.common import NormalizeConfig
from dataset import ImageDataset
from tester_s import Tester
from utility import VideoMetadata, FFmpegOption

import tensorflow as tf
tf.enable_eager_execution()

parser = argparse.ArgumentParser()

#directory, path
parser.add_argument('--dataset_dir', type=str, required=True)
parser.add_argument('--ffmpeg_path', type=str, required=True)
parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

#video metadata
parser.add_argument('--video_format', type=str, default='webm')
parser.add_argument('--start_time', type=int, default=None)
parser.add_argument('--duration', type=int, default=None)
parser.add_argument('--filter_type', type=str, choices=['uniform', 'keyframes',], default='uniform')
parser.add_argument('--filter_fps', type=float, default=1.0)
parser.add_argument('--upsample', type=str, default='bilinear')

#dataset
parser.add_argument('--input_resolution', type=int, required=True)
parser.add_argument('--target_resolution', type=int, required=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--load_on_memory', action='store_true')

#architecture
parser.add_argument('--num_filters', type=int, required=True)
parser.add_argument('--num_blocks', type=int, required=True)

#module
parser.add_argument('--model_type', type=str, required=True)

#log
parser.add_argument('--save_image', action='store_true')


args = parser.parse_args()

#0. setting
video_dir = os.path.join(args.dataset_dir, 'video')
image_dir = os.path.join(args.dataset_dir, 'image')
checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint')
log_dir = os.path.join(args.dataset_dir, 'log')
if not os.path.exists(video_dir):
    raise ValueError('directory does not exist: {}'.format(video_dir))

model_module = import_module('model.' + args.model_type)
tester_module = import_module('tester_s')
model_builder = getattr(model_module, 'model')
tester_builder = getattr(tester_module, 'Tester')

#1. creat datasets
video_metadata = VideoMetadata(args.video_format, args.start_time, args.duration)
ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
dataset = ImageDataset(video_dir,
                            image_dir,
                            video_metadata,
                            ffmpeg_option,
                            args.ffmpeg_path)

with tf.device('cpu:0'):
    print(args.load_on_memory)
    train_ds, valid_ds, rgb_mean, scale = dataset.dataset(args.input_resolution,
                                            args.target_resolution,
                                            args.batch_size,
                                            args.patch_size,
                                            args.load_on_memory)

#2. create a DNN
normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
model = model_builder(args.num_blocks, args.num_filters, scale, normalize_config)

#3. create a tester
dataset_tag = '{}.{}'.format(video_metadata.summary(args.input_resolution, True), ffmpeg_option.summary())
model_tag = '{}'.format(model.name)

checkpoint_dir = os.path.join(checkpoint_dir, dataset_tag, model_tag)
log_dir = os.path.join(log_dir, dataset_tag)
tester = tester_builder(model, checkpoint_dir, log_dir, os.path.join(image_dir, dataset_tag))

#4. train a model
tester.test(valid_ds, args.save_image)
