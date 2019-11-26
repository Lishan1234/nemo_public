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

from model.common import NormalizeConfig, LinearQuantizer
from model.edsr_ed_s import EDSR_ED_S
from dataset import valid_image_dataset, single_image_dataset, setup_images
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption, upscale_factor

tf.enable_eager_execution()

class Tester:
    def __init__(self, edsr_ed_s, quantizer, checkpoint_dir, log_dir, image_dir):
        self.lr_image_dir = lr_image_dir
        self.hr_image_dir = hr_image_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.image_dir = image_dir

        self.encode_image_dir = os.path.join(self.image_dir, 'encode')
        self.decode_image_dir = os.path.join(self.image_dir, 'decode')
        self.qnt_image_dir = os.path.join(self.image_dir, 'quantizer')
        self.quantizer = quantizer

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.qnt_image_dir, exist_ok=True)
        os.makedirs(self.decode_image_dir, exist_ok=True)
        os.makedirs(self.encode_image_dir, exist_ok=True)

        self.encoder = edsr_ed_s.load_encoder(self.checkpoint_dir)
        self.decoder = edsr_ed_s.load_decoder(self.checkpoint_dir)

    def measure_entropy(self, lr_image_dir):
        dataset = single_image_dataset(lr_image_dir)
        lr_entropy_values = []
        feature_entropy_values = []

        for idx, img in enumerate(dataset):
            now = time.perf_counter()
            lr = tf.cast(img, tf.float32)
            feature = self.encoder(lr)
            feature = self.quantizer.quantize(feature)

            lr = tf.cast(img, tf.uint8)
            feature = tf.cast(feature, tf.uint8)
            lr = lr.numpy()
            feature = feature.numpy()

            lr_gray = rgb2gray(lr)
            feature_gray= rgb2gray(feature)

            lr_entropy_value = shannon_entropy(lr_gray)
            feature_entropy_value = shannon_entropy(feature_gray)

            lr_entropy_values.append(lr_entropy_value)
            feature_entropy_values.append(feature_entropy_value)

            duration = time.perf_counter() - now
            print('lr_entropy={:.2f} feature_entropy={:.2f} (duration: {:.2f}s)'.format(lr_entropy_value, feature_entropy_value, duration))
        print('summary: lr_entropy={:.2f} feature_entropy={:.2f}'.format(np.average(lr_entropy_value), np.average(feature_entropy_value)))

        entropy_log_path = os.path.join(self.log_dir, 'entropy.txt')
        with open(entropy_log_path, 'w') as f:
            f.write('Average\t{:.2f}\t{:.2f}\n'.format(np.average(lr_entropy_values), np.average(feature_entropy_values[1])))
            for idx, entropy_values in enumerate(list(zip(lr_entropy_values, feature_entropy_values))):
                f.write('{}\t{:.2f}\t{:.2f}\n'.format(idx, entropy_values[0], entropy_values[1]))

    def test(self, lr_image_dir, hr_image_dir, save_image=False):
        #quantization
        self.quantizer.profile(self.encoder, lr_image_dir, self.checkpoint_dir)
        self.quantizer.plot(self.encoder, lr_image_dir, self.qnt_image_dir)
        self.quantizer.load(self.checkpoint_dir)

        #entropy
        self.measure_entropy(lr_image_dir)

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
            feature_qnt = self.quantizer.quantize(feature)
            feature_qnt = self.quantizer.dequantize(feature_qnt)

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
                tf.io.write_file(os.path.join(self.decode_image_dir, '{0:04d}.png'.format(idx+1)), sr_image)

            duration = time.perf_counter() - now

            print(f'PSNR(SR) = {sr_psnr_value:.3f}, PNSR(SR-Q) = {sr_qnt_psnr_value:.3f}, \
                    PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')

        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PNSE(SR-Q) = {np.average(sr_qnt_psnr_values):.3f}, \
            PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

        #log
        quality_log_path = os.path.join(self.log_dir, 'quality.txt')
        with open(quality_log_path, 'w') as f:
            f.write('Average\t{:.2f}\t\{:.2f}\t{:.2f}\n'.format(np.average(sr_psnr_values), np.average(sr_qnt_psnr_values), np.average(bilinear_psnr_values)))
            for idx, psnr_values in enumerate(list(zip(sr_psnr_values, sr_qnt_psnr_values, bilinear_psnr_values))):
                f.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(idx, psnr_values[0], psnr_values[1], psnr_values[2]))

if __name__ == '__main__':
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
    parser.add_argument('--upsample', type=str, default='bilinear')

    #architecture
    parser.add_argument('--enc_num_filters', type=int, required=True)
    parser.add_argument('--enc_num_blocks', type=int, required=True)
    parser.add_argument('--dec_num_filters', type=int, required=True)
    parser.add_argument('--dec_num_blocks', type=int, required=True)
    parser.add_argument('--enable_normalization', action='store_true')
    parser.add_argument('--min_percentile', type=float, required=True)
    parser.add_argument('--max_percentile', type=float, required=True)

    #log
    parser.add_argument('--custom_tag', type=str, default=None)
    parser.add_argument('--save_image', action='store_true')

    args = parser.parse_args()

    #setting
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))

    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #dnn
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

    #quantization
    linear_qnt = LinearQuantizer(args.min_percentile, args.max_percentile)

    #trainer
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), edsr_ed_s.name)
    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(args.lr_video_name), edsr_ed_s.name, linear_qnt.name)
    image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name), edsr_ed_s.name, linear_qnt.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    tester = Tester(edsr_ed_s, linear_qnt, checkpoint_dir, log_dir, image_dir)
    tester.test(lr_image_dir, hr_image_dir, args.save_image)
