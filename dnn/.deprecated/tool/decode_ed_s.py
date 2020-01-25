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

from model.common import NormalizeConfig, LinearQuantizer
from model.edsr_ed_s import EDSR_ED_S
from dataset import valid_feature_dataset, single_image_dataset, setup_images
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption, upscale_factor, video_fps

tf.enable_eager_execution()

class Tester:
    def __init__(self, edsr_ed_s, quantizer, checkpoint_dir, log_dir, image_dir, video_dir):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.image_dir = image_dir
        self.video_dir = video_dir

        self.decode_image_dir = os.path.join(self.image_dir, 'decode')
        self.quantizer = quantizer

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.decode_image_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        print(self.checkpoint_dir)
        self.decoder = edsr_ed_s.load_decoder(self.checkpoint_dir)

    def test(self, lr_image_dir, feature_image_dir, hr_image_dir, ffmpeg_path, output_video_name, fps, save_video, num_threads=4):
        assert(self.decoder is not None)

        #quantization
        self.quantizer.load(self.checkpoint_dir)

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

            feature = self.quantizer.dequantize(feature_qnt)
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
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--train_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

    #video metadata
    parser.add_argument('--filter_type', type=str,  default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)
    parser.add_argument('--upsample', type=str, default='bilinear')
    parser.add_argument('--bitrate', type=int, nargs='+', default=None)

    #architecture
    parser.add_argument('--enc_num_filters', type=int, required=True)
    parser.add_argument('--enc_num_blocks', type=int, required=True)
    parser.add_argument('--dec_num_filters', type=int, nargs='+', required=True)
    parser.add_argument('--dec_num_blocks', type=int, nargs='+', required=True)
    parser.add_argument('--enable_normalization', action='store_true')
    parser.add_argument('--min_percentile', type=float, required=True)
    parser.add_argument('--max_percentile', type=float, required=True)

    #log
    parser.add_argument('--custom_tag', type=str, default=None)
    parser.add_argument('--save_video', action='store_true')

    args = parser.parse_args()

    #setting (lr, hr, train)
    ffmpeg_option_0 = FFmpegOption('none', None, None) #for a pretrained DNN
    ffmpeg_option_1 = FFmpegOption(args.filter_type, args.filter_fps, args.upsample) #for a test video

    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))

    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option_0.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option_0.filter())

    scale = upscale_factor(lr_video_path, hr_video_path)

    #quantization
    linear_quantizer = LinearQuantizer(args.min_percentile, args.max_percentile)

    for dec_num_blocks in args.dec_num_blocks:
        for dec_num_filters in args.dec_num_filters:
            #dnn
            edsr_ed_s = EDSR_ED_S(args.enc_num_blocks, args.enc_num_filters, \
                          dec_num_blocks, dec_num_filters, \
                            scale, None)

            for bitrate in args.bitrate:
                #frame
                lr_video_title, lr_video_format = os.path.splitext(args.lr_video_name)
                feature_video_title = '{}_{}_encode'.format(lr_video_title, linear_quantizer.name)
                if bitrate is not 0:
                    feature_video_title += '_{}kbps'.format(bitrate)
                feature_video_name = feature_video_title + lr_video_format
                feature_video_path = os.path.join(args.dataset_dir, 'video', edsr_ed_s.name, feature_video_name)
                assert(os.path.exists(feature_video_path))
                feature_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(feature_video_name), edsr_ed_s.name)
                setup_images(feature_video_path, feature_image_dir, args.ffmpeg_path, ffmpeg_option_0.filter())

                #directory
                if args.train_video_name is not None:
                    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option_1.summary(args.train_video_name), edsr_ed_s.name)
                    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option_0.summary(feature_video_name), edsr_ed_s.name, \
                                        ffmpeg_option_1.summary(args.train_video_name))
                    image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(feature_video_name), edsr_ed_s.name, \
                                        ffmpeg_option_1.summary(args.train_video_name))
                    video_dir = os.path.join(args.dataset_dir, 'video', edsr_ed_s.name, ffmpeg_option_1.summary(args.train_video_name))
                else:
                    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option_1.summary(feature_video_name), edsr_ed_s.name)
                    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option_0.summary(feature_video_name), edsr_ed_s.name, \
                                        ffmpeg_option_1.summary(feature_video_name))
                    image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(feature_video_name), edsr_ed_s.name, \
                                        ffmpeg_option_1.summary(feature_video_name))
                    video_dir = os.path.join(args.dataset_dir, 'video', edsr_ed_s.name, ffmpeg_option_1.summary(feature_video_name))

                #video
                new_video = feature_video_name.replace('encode', 'decode')
                fps = video_fps(lr_video_path)

                #test
                tester = Tester(edsr_ed_s, linear_quantizer, checkpoint_dir, log_dir, image_dir, video_dir)
                tester.test(lr_image_dir, feature_image_dir, hr_image_dir, args.ffmpeg_path, new_video, fps, args.save_video)