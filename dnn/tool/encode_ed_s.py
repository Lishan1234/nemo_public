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
from dataset import valid_image_dataset, single_image_dataset, setup_images
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption, upscale_factor, video_fps

tf.enable_eager_execution()

class Tester:
    def __init__(self, edsr_ed_s, quantizer, checkpoint_dir, log_dir, image_dir, video_dir):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.image_dir = image_dir
        self.video_dir = video_dir

        self.encode_image_dir = os.path.join(self.image_dir, 'encode')
        self.quantizer = quantizer

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.encode_image_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        print(self.checkpoint_dir)
        self.encoder = edsr_ed_s.load_encoder(self.checkpoint_dir)

    def test(self, lr_filtered_image_dir, lr_image_dir, ffmpeg_path, output_video_name, fps, num_threads=4):
        #quantization
        self.quantizer.profile(self.encoder, lr_filtered_image_dir, self.checkpoint_dir)
        self.quantizer.load(self.checkpoint_dir)

        #quality
        lr_image_ds = single_image_dataset(lr_image_dir)
        sr_psnr_values = []
        sr_qnt_psnr_values = []
        bilinear_psnr_values = []
        for idx, img in enumerate(lr_image_ds):
            now = time.perf_counter()
            lr = tf.cast(img, tf.float32)

            feature = self.encoder(lr)
            feature_qnt = self.quantizer.quantize(feature)

            #save feature images
            feature_qnt = tf.cast(feature_qnt, tf.uint8)
            feature_qnt_image = tf.image.encode_png(tf.squeeze(feature_qnt))
            tf.io.write_file(os.path.join(self.encode_image_dir, '{0:04d}.png'.format(idx+1)), feature_qnt_image)

            duration = time.perf_counter() - now
            print(f'{idx}: ({duration:.2f}s)')

        #video
        output_video_path = os.path.join(self.video_dir, output_video_name)
        cmd = "{} -framerate {} -i {}/%04d.png -threads {} -c:v libvpx-vp9 -lossless 1 -row-mt 1 -c:a libopus {}".format(ffmpeg_path, fps, self.encode_image_dir, num_threads, output_video_path)
        print(cmd)
        os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

    #video metadata
    parser.add_argument('--filter_type', type=str,  default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)
    parser.add_argument('--upsample', type=str, default='bilinear')

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

    args = parser.parse_args()


    #setting
    ffmpeg_option_0 = FFmpegOption('none', None, None) #for a pretrained DNN
    ffmpeg_option_1 = FFmpegOption(args.filter_type, args.filter_fps, args.upsample) #for a test video

    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))
    scale = upscale_factor(lr_video_path, hr_video_path)

    lr_filtered_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_1.summary(args.lr_video_name))
    setup_images(lr_video_path, lr_filtered_image_dir, args.ffmpeg_path, ffmpeg_option_1.filter())

    #quantization
    linear_quantizer = LinearQuantizer(args.min_percentile, args.max_percentile)

    for dec_num_blocks in args.dec_num_blocks:
        for dec_num_filters in args.dec_num_filters:
            #dnn
            edsr_ed_s = EDSR_ED_S(args.enc_num_blocks, args.enc_num_filters, \
                                   dec_num_blocks, dec_num_filters, \
                                    scale, None)

            #frame
            lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(args.lr_video_name))
            setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option_0.filter())

            #directory
            checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option_1.summary(args.lr_video_name), edsr_ed_s.name)
            log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option_0.summary(args.lr_video_name), edsr_ed_s.name, \
                                    linear_quantizer.name)
            image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_0.summary(args.lr_video_name), edsr_ed_s.name, \
                                    linear_quantizer.name)
            video_dir = os.path.join(args.dataset_dir, 'video', edsr_ed_s.name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            #video
            video_name, video_format = os.path.splitext(args.lr_video_name)
            new_video = '{}_{}_encode{}'.format(video_name, linear_quantizer.name, video_format)
            fps = video_fps(lr_video_path)

            #test
            tester = Tester(edsr_ed_s, linear_quantizer, checkpoint_dir, log_dir, image_dir, video_dir)
            tester.test(lr_filtered_image_dir, lr_image_dir, args.ffmpeg_path, new_video, fps)
