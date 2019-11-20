import time
import os
import sys
import argparse
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from model.common import NormalizeConfig, QuantizeConfig, quantize, dequantize
from model.edsr_ed_s import EDSR_ED_S
from dataset import valid_image_dataset, single_image_dataset, setup_images
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption, upscale_factor, measure_entropy

tf.enable_eager_execution()

class Tester:
    def __init__(self, edsr_ed_s, quantization_policy, checkpoint_dir, log_dir, image_dir):
        self.lr_image_dir = lr_image_dir
        self.hr_image_dir = hr_image_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.image_dir = image_dir

        self.encode_image_dir = os.path.join(self.image_dir, 'encode')
        self.decode_image_dir = os.path.join(self.image_dir, 'decode')
        self.qnt_image_dir = os.path.join(self.image_dir, 'quantization')
        self.qnt_config = QuantizeConfig(quantization_policy)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.qnt_image_dir, exist_ok=True)
        os.makedirs(self.decode_image_dir, exist_ok=True)
        os.makedirs(self.encode_image_dir, exist_ok=True)

        self.encoder = edsr_ed_s.load_encoder(self.checkpoint_dir)
        self.decoder = edsr_ed_s.load_decoder(self.checkpoint_dir)

    def test(self, lr_image_dir, hr_image_dir, save_image=False):
        #quantization
        lr_image_ds = single_image_dataset(lr_image_dir)
        self.qnt_config.set(self.encoder, lr_image_ds, self.checkpoint_dir, self.qnt_image_dir)
        print('quantization: min({:.2f}) max({:.2f})'.format(self.qnt_config.enc_min, self.qnt_config.enc_max))

        #entropy
        lr_image_ds = single_image_dataset(lr_image_dir)
        lr_entropy_values, feature_entropy_values = measure_entropy(self.encoder, lr_image_ds, self.qnt_config)
        entropy_log_path = os.path.join(self.log_dir, 'entropy.txt')
        with open(entropy_log_path, 'w') as f:
            for entropy_values in list(zip(lr_entropy_values, feature_entropy_values)):
                f.write('{:.2f}\t{:.2f}\n'.format(entropy_values[0], entropy_values[1]))

        #quality
        valid_image_ds = valid_image_dataset(lr_image_dir, hr_image_dir)
        sr_psnr_values = []
        sr_qnt_psnr_values = []
        bilinear_psnr_values = []
        for idx, imgs in enumerate(valid_image_ds):
            now = time.perf_counter()
            lr = tf.cast(imgs[0], tf.float32)
            hr = imgs[1]

            #measure height, width
            if idx == 0:
                hr_shape = tf.shape(hr)[1:3]
                height = hr_shape[0].numpy()
                width = hr_shape[1].numpy()

            feature = self.encoder(lr)
            feature_qnt = quantize(feature, self.qnt_config.enc_min, self.qnt_config.enc_max)
            feature_qnt = dequantize(feature_qnt, self.qnt_config.enc_min, self.qnt_config.enc_max)

            sr = self.decoder(feature)
            sr_qnt = self.decoder(feature_qnt)

            #measure sr quality
            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)
            sr_psnr_value = tf.image.psnr(hr, sr, max_val=255)[0].numpy()
            sr_psnr_values.append(sr_psnr_value)

            #measure (quantized) sr quality
            sr_qnt = tf.clip_by_value(sr_qnt, 0, 255)
            sr_qnt = tf.round(sr_qnt)
            sr_qnt = tf.cast(sr_qnt, tf.uint8)
            sr_qnt_psnr_value = tf.image.psnr(hr, sr_qnt, max_val=255)[0].numpy()
            sr_qnt_psnr_values.append(sr_qnt_psnr_value)

            #measure bilinear quality
            lr = tf.cast(lr, tf.float32)
            bilinear = resolve_bilinear(lr, height, width)
            bilinear = tf.cast(bilinear, tf.uint8)
            bilinear_psnr_value = tf.image.psnr(hr, bilinear, max_val=255)[0].numpy()
            bilinear_psnr_values.append(bilinear_psnr_value)

            #save sr images
            if save_image:
                sr_image = tf.image.encode_png(tf.squeeze(sr))
                tf.io.write_file(os.path.join(self.decode_image_dir, '{0:04d}.png'.format(idx)), sr_image)

            duration = time.perf_counter() - now

            print(f'PSNR(SR) = {sr_psnr_value:.3f}, PNSR(SR-Q) = {sr_qnt_psnr_value:.3f}, \
                    PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')

        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PNSE(SR-Q) = {np.average(sr_qnt_psnr_values):.3f}, \
            PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

        quality_log_path = os.path.join(self.log_dir, 'quality.txt')
        with open(quality_log_path, 'w') as f:
            for psnr_values in list(zip(sr_psnr_values, sr_qnt_psnr_values, bilinear_psnr_values)):
                f.write('{:.2f}\t{:.2f}\t{:.2f}\n'.format(psnr_values[0], psnr_values[1], psnr_values[2]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

    #video metadata
    parser.add_argument('--filter_type', type=str, choices=['uniform', 'keyframes',], default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)
    parser.add_argument('--upsample', type=str, default='bilinear')

    #architecture
    parser.add_argument('--enc_num_filters', type=int, required=True)
    parser.add_argument('--enc_num_blocks', type=int, required=True)
    parser.add_argument('--dec_num_filters', type=int, required=True)
    parser.add_argument('--dec_num_blocks', type=int, required=True)
    parser.add_argument('--enable_normalization', action='store_true')
    parser.add_argument('--quantization_policy', type=str, required=True)

    #log
    parser.add_argument('--custom_tag', type=str, default=None)
    parser.add_argument('--save_image', action='store_ture')

    args = parser.parse_args()

    #0. setting
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))

    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #1. dnn
    scale = upscale_factor(lr_video_path, hr_video_path)
    if args.enable_normalization:
        #TODO: rgb mean
        #normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
        pass
    else:
        normalize_config = None
    edsr_ed_s = EDSR_ED_S(args.enc_num_blocks, args.enc_num_filters, \
                           args.dec_num_blocks, args.dec_num_filters, \
                            scale, normalize_config)

    #2. create a trainer
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), edsr_ed_s.name)
    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(args.lr_video_name), edsr_ed_s.name)
    image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name), edsr_ed_s.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    tester = Tester(edsr_ed_s, args.quantization_policy, checkpoint_dir, log_dir, image_dir)
    tester.test(lr_image_dir, hr_image_dir, args.save_image)
