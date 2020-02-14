#Training script for a single resolution network

import argparse
import os
import sys

from model.edsr_r import EDSR_S
from dataset import ImageDataset
from trainer_s import EDSRTrainer
from utility import VideoMetadata, FFmpegOption

import tensorflow as tf

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

args = parser.parse_args()

if tf.executing_eagerly():
    raise RuntimeError('Eager mode should be turn-off')

#0. setting
video_dir = os.path.join(args.dataset_dir, 'video')
image_dir = os.path.join(args.dataset_dir, 'image')
checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint')
log_dir = os.path.join(args.dataset_dir, 'log')
if not os.path.exists(video_dir):
    raise ValueError('directory does not exist: {}'.format(video_dir))

edsr_s = EDSR_S(args.num_blocks, args.num_filters, args.scale)

#1. get scale
video_metadata = VideoMetadata(args.video_format, args.start_time, args.duration)
ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
dataset = ImageDataset(video_dir,
                            image_dir,
                            video_metadata,
                            ffmpeg_option,
                            args.ffmpeg_path)

with tf.device('cpu:0'):
    train_ds, valid_ds, train_dir, valid_dir, rgb_mean, scale = dataset.dataset(args.input_resolution,
                                            args.target_resolution,
                                            args.batch_size,
                                            args.patch_size,
                                            False)

#2. create a DNN
#normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
with tf.Graph().as_default(), tf.Session() as sess:
    init = tf.global_variables_initializer()
    #model = model_builder(args.num_blocks, args.num_filters, scale, normalize_config)
    model = model_builder(args.num_blocks, args.num_filters, scale, None)
    sess.run(init)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
