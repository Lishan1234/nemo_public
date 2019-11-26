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
from dataset import valid_feature_dataset, single_image_dataset, setup_images
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption, upscale_factor, measure_entropy, video_fps

tf.enable_eager_execution()

class Tester:
    def __init__(self, edsr_ed_s, quantization_policy, checkpoint_dir, log_dir, image_dir, video_dir):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.image_dir = image_dir
        self.video_dir = video_dir

        self.decode_image_dir = os.path.join(self.image_dir, 'decode')
        self.qnt_config = QuantizeConfig(quantization_policy)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.decode_image_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        print(self.checkpoint_dir)
        self.decoder = edsr_ed_s.load_decoder(self.checkpoint_dir)

    def test(self, lr_image_dir, feature_image_dir, hr_image_dir, ffmpeg_path, output_video_name, fps, save_video, num_threads=4):
        assert(self.decoder is not None)

        #quantization
        self.qnt_config.load(self.checkpoint_dir)

        #quality
        valid_feature_ds = valid_feature_dataset(lr_image_dir, feature_image_dir, hr_image_dir)
        sr_psnr_values = []
        bilinear_psnr_values = []
        for idx, imgs in enumerate(valid_feature_ds):
            now = time.perf_counter()

            lr = imgs[0]
            feature_qnt = tf.cast(imgs[1], tf.float32)
            hr = imgs[2]

            #measure height, width
            if idx == 0:
                hr_shape = tf.shape(hr)[1:3]
                height = hr_shape[0].numpy()
                width = hr_shape[1].numpy()

            feature = dequantize(feature_qnt, self.qnt_config.enc_min, self.qnt_config.enc_max)
            sr = self.decoder(feature)

            #measure sr quality
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

            #save sr images
            if save_video:
                sr_image = tf.image.encode_png(tf.squeeze(sr))
                tf.io.write_file(os.path.join(self.decode_image_dir, '{0:04d}.png'.format(idx+1)), sr_image)

            duration = time.perf_counter() - now
            print(f'PSNR(SR) = {sr_psnr_value:.3f}, PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')
        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

        #log
        quality_log_path = os.path.join(self.log_dir, 'quality.txt')
        with open(quality_log_path, 'w') as f:
            f.write('Average\t{:.2f}\t{:.2f}\n'.format(np.average(sr_psnr_values), np.average(bilinear_psnr_values)))
            for idx, psnr_values in enumerate(list(zip(sr_psnr_values, bilinear_psnr_values))):
                f.write('{}\t{:.2f}\t{:.2f}\n'.format(idx, psnr_values[0], psnr_values[1]))

        #video
        if save_video:
            output_video_path = os.path.join(self.video_dir, output_video_name)
            cmd = "{} -framerate {} -i {}/%04d.png -threads {} -c:v libvpx-vp9 -lossless 1 -row-mt 1 -c:a libopus {}".format(ffmpeg_path, fps, self.decode_image_dir, num_threads, output_video_path)
            os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--feature_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--train_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

    #video metadata
    parser.add_argument('--filter_type', type=str,  default='uniform')
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
    parser.add_argument('--save_video', action='store_true')

    args = parser.parse_args()

    assert('encode' in args.feature_video_name)

    #setting (lr, hr, train)
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    train_video_path = os.path.join(args.dataset_dir, 'video', args.train_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))
    assert(os.path.exists(train_video_path))

    ffmpeg_option_0 = FFmpegOption('none', None, None) #for a pretrained DNN
    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option_0.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option_0.filter())

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

    #setting (feature)
    ffmpeg_option_1 = FFmpegOption(args.filter_type, args.filter_fps, args.upsample) #for a test video
    feature_video_path = os.path.join(args.dataset_dir, 'video', edsr_ed_s.name, args.feature_video_name)
    assert(os.path.exists(feature_video_path))
    feature_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(args.feature_video_name), edsr_ed_s.name)
    setup_images(feature_video_path, feature_image_dir, args.ffmpeg_path, ffmpeg_option_0.filter())

    #tester
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option_1.summary(args.train_video_name), edsr_ed_s.name)
    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option_0.summary(args.feature_video_name), edsr_ed_s.name, \
                        ffmpeg_option_1.summary(args.train_video_name))
    image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(args.feature_video_name), edsr_ed_s.name, \
                        ffmpeg_option_1.summary(args.train_video_name))
    video_dir = os.path.join(args.dataset_dir, 'video', edsr_ed_s.name, ffmpeg_option_1.summary(args.train_video_name))

    new_video = args.feature_video_name.replace('encode', 'decode')
    fps = video_fps(lr_video_path)

    tester = Tester(edsr_ed_s, args.quantization_policy, checkpoint_dir, log_dir, image_dir, video_dir)
    tester.test(lr_image_dir, feature_image_dir, hr_image_dir, args.ffmpeg_path, new_video, fps, args.save_video)
