from importlib import import_module
import time
import os
import sys
import argparse
import math

import numpy as np
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from model.common import NormalizeConfig, QuantizeConfig, quantize, dequantize
from model.edsr_ed_s import EDSR_ED_S
from dataset import valid_image_dataset, single_image_dataset, setup_images
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption

import tensorflow as tf
tf.enable_eager_execution()

class Tester:
    decode_log_name = 'quality.txt'
    quantization_subdir = 'quantization'
    decode_subdir = 'decode'
    encode_subdir = 'encode'

    def __init__(self, edsr_ed_s, quantize_config, checkpoint_dir, log_dir, lr_image_dir):
        self.edsr_ed_s = edsr_ed_s
        self.lr_image_dir = lr_image_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = os.path.join(log_dir, self.edsr_ed_s.name)

        self.encoder = edsr_ed_s.load_encoder(self.checkpoint_dir)
        self.decoder = edsr_ed_s.load_decoder(self.checkpoint_dir)
        self.encode_image_dir = os.path.join(lr_image_dir, self.encode_subdir)
        self.qnt_config = quantize_config

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.encode_image_dir, exist_ok=True)

    def test(self, save_image=False):
        #quantization
        self.qnt_config.load(self.checkpoint_dir)
        print('quantization: min({:.2f}) max({:.2f})'.format(self.qnt_config.enc_min, self.qnt_config.enc_max))

        #quality
        lr_image_ds = single_image_dataset(self.lr_image_dir)
        for idx, imgs in enumerate(lr_image_ds):
            now = time.perf_counter()
            lr = tf.cast(imgs, tf.float32)
            feature = self.encoder(lr)
            feature_qnt = quantize(feature, qnt_config.enc_min, qnt_config.enc_max)
            feature_qnt = tf.cast(feature_qnt, tf.uint8)

            #save sr images
            if save_image:
                feature_image = tf.image.encode_png(tf.squeeze(feature_qnt))
                tf.io.write_file(os.path.join(self.encode_image_dir, '{0:04d}.png'.format(idx)), feature_image)

            duration = time.perf_counter() - now
            print(f'{idx}: {duration:.2f}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

    #video metadata
    parser.add_argument('--video_format', type=str, default='webm')
    parser.add_argument('--start_time', type=int, default=None)
    parser.add_argument('--duration', type=int, default=None)
    parser.add_argument('--filter_type', type=str, default='uniform')
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
    parser.add_argument('--enable_normalization', action='store_true')
    parser.add_argument('--quantization_policy', type=str, default='min_max_0')

    #log
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--custom_tag', type=str, default=None)

    args = parser.parse_args()

    #setting
    video_metadata = VideoMetadata(args.video_format, args.start_time, args.duration)
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
    lr_video_name = video_metadata.summary(args.input_resolution, True)
    lr_video_path = os.path.join(args.dataset_dir, 'video', lr_video_name)
    assert(os.path.exists(lr_video_path))

    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(lr_video_name))
    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(lr_video_name))
    os.makedirs(log_dir, exist_ok=True)

    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #dnn
    scale = math.floor(args.target_resolution / args.input_resolution)
    if args.checkpoint_dir is None:
        checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(lr_video_name))
    else:
        checkpoint_dir = args.checkpoint_dir
    if args.enable_normalization:
        #TODO: rgb mean
        #normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
        pass
    else:
        normalize_config = None
    edsr_ed_s = EDSR_ED_S(args.enc_num_blocks, args.enc_num_filters, \
                           args.dec_num_blocks, args.dec_num_filters, \
                            scale, normalize_config)

    #test
    qnt_config = QuantizeConfig(args.quantization_policy)
    tester = Tester(edsr_ed_s, qnt_config, checkpoint_dir, log_dir, lr_image_dir)
    tester.test(True)
