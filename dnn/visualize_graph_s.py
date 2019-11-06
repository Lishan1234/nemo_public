#Training script for a single resolution network

import argparse
import os
import sys
from importlib import import_module

from model.common import NormalizeConfig
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
parser.add_argument('--trainer_type', type=str, required=True)

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

model_module = import_module('model.' + args.model_type)
trainer_module = import_module('trainer_s')
model_builder = getattr(model_module, 'model')
trainer_builder = getattr(trainer_module, args.trainer_type)

#1. get scale
scale = ImageDataset.scale(args.input_resolution, args.target_resolution)

#2. create a DNN
with tf.Graph().as_default(), tf.Session() as sess:
    init = tf.global_variables_initializer()
    model = model_builder(args.num_blocks, args.num_filters, scale, None)
    sess.run(init)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
