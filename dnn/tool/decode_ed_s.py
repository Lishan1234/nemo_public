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
from dataset import valid_image_dataset, single_image_dataset, setup_images, valid_feature_dataset
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption

import tensorflow as tf
tf.enable_eager_execution()

#TODO: write a log

class Tester:
    decode_log_name = 'quality.txt'
    decode_subdir = 'decode'

    def __init__(self, edsr_ed_s, quantize_config, checkpoint_dir, log_dir, lr_image_dir, feature_image_dir, hr_image_dir):
        self.edsr_ed_s = edsr_ed_s
        self.lr_image_dir = lr_image_dir
        self.feature_image_dir = feature_image_dir
        self.hr_image_dir = hr_image_dir
        self.checkpoint_dir = checkpoint_dir
        print(self.feature_image_dir)
        print(os.path.basename(self.feature_image_dir))
        self.log_dir = os.path.join(log_dir, self.edsr_ed_s.name, os.path.basename(self.feature_image_dir))

        self.encoder = edsr_ed_s.load_encoder(self.checkpoint_dir)
        self.decoder = edsr_ed_s.load_decoder(self.checkpoint_dir)
        self.decode_image_dir = os.path.join(self.feature_image_dir, self.decode_subdir)
        self.qnt_config = quantize_config

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.decode_image_dir, exist_ok=True)

    def test(self, save_image=False):
        #quantization
        self.qnt_config.load(self.checkpoint_dir)
        print('quantization: min({:.2f}) max({:.2f})'.format(self.qnt_config.enc_min, self.qnt_config.enc_max))

        #quality
        sr_psnr_values = []
        bilinear_psnr_values = []
        valid_feature_ds = valid_feature_dataset(self.lr_image_dir, self.feature_image_dir, self.hr_image_dir)
        log_path = os.path.join(self.log_dir, self.decode_log_name)
        with open(log_path, 'w') as f:
            for idx, imgs in enumerate(valid_feature_ds):
                now = time.perf_counter()
                lr = imgs[0]
                feature_qnt = imgs[1]
                hr = imgs[2]

                #measure height, width
                if idx == 0:
                    hr_shape = tf.shape(hr)[1:3]
                    height = hr_shape[0].numpy()
                    width = hr_shape[1].numpy()

                #measure decoder quality
                feature_qnt = tf.cast(feature_qnt, tf.float32)
                feature = dequantize(feature_qnt, qnt_config.enc_min, qnt_config.enc_max)
                sr = self.decoder(feature)
                sr = tf.clip_by_value(sr, 0, 255)
                sr = tf.round(sr)
                sr = tf.cast(sr, tf.uint8)
                sr_psnr_value = tf.image.psnr(hr, sr, max_val=255)[0].numpy()
                sr_psnr_values.append(sr_psnr_value)

                #measure bilinear quality
                lr = tf.cast(lr, tf.float32)
                bilinear = resolve_bilinear(lr, height, width)
                bilinear = tf.cast(bilinear, tf.uint8)
                bilinear_psnr_value = tf.image.psnr(hr, bilinear, max_val=255)[0].numpy()
                bilinear_psnr_values.append(bilinear_psnr_value)

                #log
                f.write('{:.2f}\t{:.2f}\n'.format(sr_psnr_value, bilinear_psnr_value))

                #save sr images
                if save_image:
                    sr_image = tf.image.encode_png(tf.squeeze(sr))
                    tf.io.write_file(os.path.join(self.decode_image_dir, '{0:04d}.png'.format(idx)), sr_image)

                duration = time.perf_counter() - now
                print(f'PSNR(SR) = {sr_psnr_value:.3f}, PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')
        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--feature_image_dir', type=str, default=None)
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
    hr_video_name = video_metadata.summary(args.target_resolution, False)
    lr_video_path = os.path.join(args.dataset_dir, 'video', lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', hr_video_name)
    assert(os.path.exists(lr_video_path))
    lr_video_path = os.path.join(args.dataset_dir, 'video', lr_video_name)

    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(lr_video_name))
    feature_image_dir = args.feature_image_dir
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(hr_video_name))
    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(lr_video_name))
    os.makedirs(log_dir, exist_ok=True)

    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

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
    tester = Tester(edsr_ed_s, qnt_config, checkpoint_dir, log_dir, lr_image_dir, feature_image_dir, hr_image_dir)
    tester.test(True)
