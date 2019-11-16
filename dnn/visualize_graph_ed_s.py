#Training script for a single resolution network

import argparse
import os
import sys
from importlib import import_module

from model.common import NormalizeConfig
from model.edsr_ed_s import EDSR_ED_S
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
parser.add_argument('--enc_num_filters', type=int, required=True)
parser.add_argument('--enc_num_blocks', type=int, required=True)
parser.add_argument('--dec_num_filters', type=int, required=True)
parser.add_argument('--dec_num_blocks', type=int, required=True)

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

#1. get scale
video_metadata = VideoMetadata(args.video_format, args.start_time, args.duration)
ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
dataset = ImageDataset(video_dir,
                            image_dir,
                            video_metadata,
                            ffmpeg_option,
                            args.ffmpeg_path)

with tf.device('cpu:0'):
    scale = dataset.scale(args.input_resolution, args.target_resolution)

#2. create a DNN
#normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
edsr_ed_s = EDSR_ED_S(args.enc_num_blocks, args.enc_num_filters, \
                        args.dec_num_blocks, args.dec_num_filters, scale, None)
with tf.Graph().as_default(), tf.Session() as sess:
    init = tf.global_variables_initializer()
    model = edsr_ed_s.build_model()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
