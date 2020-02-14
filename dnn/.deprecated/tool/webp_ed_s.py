import time
import os
import sys
import argparse
import math
import glob
import subprocess
import shlex

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from model.common import NormalizeConfig, LinearQuantizer
from model.edsr_ed_s import EDSR_ED_S
from dataset import valid_feature_dataset, single_image_dataset, setup_images, valid_image_dataset
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption, upscale_factor, video_fps

tf.enable_eager_execution()

class Tester:
    near_lossless = [0, 20, 40, 60, 80, 100]
    q = [0, 20, 40, 60, 80, 100]

    def __init__(self, edsr_ed_s, quantizer, checkpoint_dir, log_dir, image_dir, webp_dir):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.image_dir = image_dir
        self.webp_dir = webp_dir
        self.quantizer = quantizer
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        self.decoder = edsr_ed_s.load_decoder(self.checkpoint_dir)
        self.quantizer.load(self.checkpoint_dir)

    def quality(self, lr_image_dir, feature_image_dir, decode_image_dir, hr_image_dir, log_dir):
        #lr
        valid_image_ds = valid_image_dataset(feature_image_dir, decode_image_dir)
        feature_psnr_values = []
        for idx, imgs in enumerate(valid_image_ds):
            now = time.perf_counter()

            feature_raw = imgs[0]
            feature_compress = imgs[1]

            feature_psnr_value = tf.image.psnr(feature_raw, feature_compress, max_val=255)[0].numpy()
            feature_psnr_values.append(feature_psnr_value)

            duration = time.perf_counter() - now
            print(f'PSNR(feature) = {feature_psnr_value:3f} ({duration:.2f}s)')
        print(f'PSNR(feature) = {np.average(feature_psnr_values):3f}')

        #sr
        valid_feature_ds = valid_feature_dataset(lr_image_dir, decode_image_dir, hr_image_dir)
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

            #measure feature quality

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

            duration = time.perf_counter() - now
            print(f'PSNR(SR) = {sr_psnr_value:.3f}, PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')
        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

        #log
        quality_log_path = os.path.join(log_dir, 'quality.txt')
        with open(quality_log_path, 'w') as f:
            f.write('Average\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(np.average(sr_psnr_values), np.average(bilinear_psnr_values), np.average(feature_psnr_values)))
            for idx, psnr_values in enumerate(list(zip(sr_psnr_values, bilinear_psnr_values, feature_psnr_values))):
                f.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(idx, psnr_values[0], psnr_values[1], psnr_values[2]))

    def size(self, encode_image_dir, log_dir):
        encode_image_files = sorted(glob.glob('{}/*.webp'.format(encode_image_dir)))

        #measure size
        encode_image_sizes = []
        for encode_image_file in encode_image_files:
            now = time.perf_counter()

            encode_image_size = os.path.getsize(encode_image_file)
            encode_image_sizes.append(encode_image_size)

            duration = time.perf_counter() - now
            print(f'Image size = {int(encode_image_size / 1000)}KB ({duration:.2f}s)')
        print(f'Summary: Image size = {int(np.average(encode_image_sizes) / 1000)}KB')

        #log
        size_log_path = os.path.join(log_dir, 'size.txt')
        with open(size_log_path, 'w') as f:
            f.write('Average\t{}\n'.format(int(np.average(encode_image_sizes) / 1000)))
            for idx, encode_image_size in enumerate(encode_image_sizes):
                f.write('{}\t{}\n'.format(idx, encode_image_size))

    def test_lossless(self, lr_image_dir, feature_image_dir, hr_image_dir):
        feature_image_files = sorted(glob.glob('{}/*.png'.format(feature_image_dir)))

        print('start: lossless')
        encode_image_dir = os.path.join(self.image_dir, 'webp_lossless', 'encode')
        decode_image_dir = os.path.join(self.image_dir, 'webp_lossless', 'decode')
        log_dir = os.path.join(self.log_dir, 'webp_lossless')
        os.makedirs(encode_image_dir, exist_ok=True)
        os.makedirs(decode_image_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        #encode,decode
        for idx, feature_image_file in enumerate(feature_image_files):
            now = time.perf_counter()

            feature_image_title, feature_image_format = os.path.splitext(os.path.basename(feature_image_file))
            encode_image_file = os.path.join(encode_image_dir, '{}.webp'.format(feature_image_title))
            enc_cmd = '{}/bin/cwebp -quiet -mt -lossless -q 100 {} -o {}'.format(self.webp_dir, feature_image_file, encode_image_file)
            os.system(enc_cmd)

            decode_image_file = os.path.join(decode_image_dir, '{}.png'.format(feature_image_title))
            dec_cmd = '{}/bin/dwebp -quiet -mt {} -o {}'.format(self.webp_dir, encode_image_file, decode_image_file)
            os.system(dec_cmd)

            duration = time.perf_counter() - now
            print(f'{idx+1}/{len(feature_image_files)} ({duration:.2f}s)')

        #quality
        self.quality(lr_image_dir, feature_image_dir, decode_image_dir, hr_image_dir, log_dir)

        #size
        self.size(encode_image_dir, log_dir)
        print('end: lossless')

    def test_near_lossless(self, lr_image_dir, feature_image_dir, hr_image_dir):
        feature_image_files = sorted(glob.glob('{}/*.png'.format(feature_image_dir)))

        for near_lossless in self.near_lossless:
            print('start: near_lossless={}'.format(near_lossless))
            encode_image_dir = os.path.join(self.image_dir, 'webp_near_lossless_{}'.format(near_lossless), 'encode')
            decode_image_dir = os.path.join(self.image_dir, 'webp_near_lossless_{}'.format(near_lossless), 'decode')
            log_dir = os.path.join(self.log_dir, 'webp_near_lossless_{}'.format(near_lossless))
            os.makedirs(encode_image_dir, exist_ok=True)
            os.makedirs(decode_image_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            #encode,decode
            for idx, feature_image_file in enumerate(feature_image_files):
                now = time.perf_counter()

                feature_image_title, feature_image_format = os.path.splitext(os.path.basename(feature_image_file))
                encode_image_file = os.path.join(encode_image_dir, '{}.webp'.format(feature_image_title))
                enc_cmd = '{}/bin/cwebp -quiet -mt -near_lossless {} {} -o {}'.format(self.webp_dir, near_lossless, feature_image_file, encode_image_file)
                os.system(enc_cmd)

                decode_image_file = os.path.join(decode_image_dir, '{}.png'.format(feature_image_title))
                dec_cmd = '{}/bin/dwebp -quiet -mt {} -o {}'.format(self.webp_dir, encode_image_file, decode_image_file)
                os.system(dec_cmd)

                duration = time.perf_counter() - now
                print(f'{idx+1}/{len(feature_image_files)} ({duration:.2f}s)')

            #quality
            self.quality(lr_image_dir, feature_image_dir, decode_image_dir, hr_image_dir, log_dir)

            #size
            self.size(encode_image_dir, log_dir)
            print('end: near_lossless={}'.format(near_lossless))

    def test_lossy(self, lr_image_dir, feature_image_dir, hr_image_dir):
        feature_image_files = sorted(glob.glob('{}/*.png'.format(feature_image_dir)))

        for q in self.q:
            print('start: lossy q={}'.format(q))
            encode_image_dir = os.path.join(self.image_dir, 'webp_lossy_{}'.format(q), 'encode')
            decode_image_dir = os.path.join(self.image_dir, 'webp_lossy_{}'.format(q), 'decode')
            log_dir = os.path.join(self.log_dir, 'webp_lossy_{}'.format(q))
            os.makedirs(encode_image_dir, exist_ok=True)
            os.makedirs(decode_image_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            #encode,decode
            for idx, feature_image_file in enumerate(feature_image_files):
                now = time.perf_counter()

                feature_image_title, feature_image_format = os.path.splitext(os.path.basename(feature_image_file))
                encode_image_file = os.path.join(encode_image_dir, '{}.webp'.format(feature_image_title))
                enc_cmd = '{}/bin/cwebp -quiet -mt -q {} {} -o {}'.format(self.webp_dir, q, feature_image_file, encode_image_file)
                os.system(enc_cmd)

                decode_image_file = os.path.join(decode_image_dir, '{}.png'.format(feature_image_title))
                dec_cmd = '{}/bin/dwebp -quiet -mt {} -o {}'.format(self.webp_dir, encode_image_file, decode_image_file)
                os.system(dec_cmd)

                duration = time.perf_counter() - now
                print(f'{idx+1}/{len(feature_image_files)} ({duration:.2f}s)')

            #quality
            self.quality(lr_image_dir, feature_image_dir, decode_image_dir, hr_image_dir, log_dir)

            #size
            self.size(encode_image_dir, log_dir)
            print('end: lossy q={}'.format(q))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--webp_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
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
    ffmpeg_option_1 = FFmpegOption(args.filter_type, args.filter_fps, args.upsample) #for a test video

    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))

    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_1.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_1.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option_1.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option_1.filter())

    scale = upscale_factor(lr_video_path, hr_video_path)

    #quantization
    linear_quantizer = LinearQuantizer(args.min_percentile, args.max_percentile)

    for dec_num_blocks in args.dec_num_blocks:
        for dec_num_filters in args.dec_num_filters:
            #dnn
            edsr_ed_s = EDSR_ED_S(args.enc_num_blocks, args.enc_num_filters, \
                          dec_num_blocks, dec_num_filters, \
                            scale, None)

            #frame
            lr_video_title, lr_video_format = os.path.splitext(args.lr_video_name)
            feature_video_title = '{}_{}_encode'.format(lr_video_title, linear_quantizer.name)
            feature_video_name = feature_video_title + lr_video_format
            feature_video_path = os.path.join(args.dataset_dir, 'video', edsr_ed_s.name, feature_video_name)
            assert(os.path.exists(feature_video_path))
            feature_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_1.summary(feature_video_name), edsr_ed_s.name)
            setup_images(feature_video_path, feature_image_dir, args.ffmpeg_path, ffmpeg_option_1.filter())

            #directory
            if args.train_video_name is not None:
                checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option_1.summary(args.train_video_name), edsr_ed_s.name)
                log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option_1.summary(feature_video_name), edsr_ed_s.name, \
                                    ffmpeg_option_1.summary(args.train_video_name))
                image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_1.summary(feature_video_name), edsr_ed_s.name, \
                                    ffmpeg_option_1.summary(args.train_video_name))
            else:
                checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option_1.summary(feature_video_name), edsr_ed_s.name)
                log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option_1.summary(feature_video_name), edsr_ed_s.name, \
                                    ffmpeg_option_1.summary(feature_video_name))
                image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option_1.summary(feature_video_name), edsr_ed_s.name, \
                                    ffmpeg_option_1.summary(feature_video_name))

            #test
            tester = Tester(edsr_ed_s, linear_quantizer, checkpoint_dir, log_dir, image_dir, args.webp_dir)
            #tester.test_lossless(lr_image_dir, feature_image_dir, hr_image_dir)
            #tester.test_near_lossless(lr_image_dir, feature_image_dir, hr_image_dir)
            #tester.test_lossy(lr_image_dir, feature_image_dir, hr_image_dir)

    result_dir = os.path.join(args.dataset_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, 'summary_webp_ed_s.txt')
    with open(result_path, 'w') as f:
        f.write('#Block\t#Filter\tMode\tCompress factor\tPSNR(SR)\tPSNR(Bilinear)\tPSNR(Feature)\tSize(KB)\n')
        for dec_num_blocks in args.dec_num_blocks:
            for dec_num_filters in args.dec_num_filters:
                edsr_ed_s = EDSR_ED_S(args.enc_num_blocks, args.enc_num_filters, \
                              dec_num_blocks, dec_num_filters, \
                                scale, None)

                if args.train_video_name is not None:
                    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option_1.summary(feature_video_name), edsr_ed_s.name, \
                                        ffmpeg_option_1.summary(args.train_video_name))
                else:
                    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option_1.summary(feature_video_name), edsr_ed_s.name, \
                                        ffmpeg_option_1.summary(feature_video_name))
                lossless_log_dir = os.path.join(log_dir, 'webp_lossless')
                quality_log_path = os.path.join(lossless_log_dir, 'quality.txt')
                size_log_path = os.path.join(lossless_log_dir, 'size.txt')
                with open(quality_log_path) as f_q, open(size_log_path) as f_s:
                    result = f_q.readline().strip().split('\t')
                    sr_quality = np.round(float(result[1]), 2)
                    bilinear_quality = np.round(float(result[2]), 2)
                    feature_quality = np.round(float(result[3]), 2)
                    size = int(f_s.readline().strip().split('\t')[1])
                    f.write('{}\t{}\tLossless\t100\t{}\t{}\t{}\t{}\n'.format(dec_num_blocks, dec_num_filters, sr_quality, bilinear_quality, feature_quality, size))

                for near_lossless in Tester.near_lossless:
                    near_lossless_log_dir = os.path.join(log_dir, 'webp_near_lossless_{}'.format(near_lossless))
                    quality_log_path = os.path.join(near_lossless_log_dir, 'quality.txt')
                    size_log_path = os.path.join(near_lossless_log_dir, 'size.txt')
                    with open(quality_log_path) as f_q, open(size_log_path) as f_s:
                        result = f_q.readline().strip().split('\t')
                        sr_quality = np.round(float(result[1]), 2)
                        bilinear_quality = np.round(float(result[2]), 2)
                        feature_quality = np.round(float(result[3]), 2)
                        size = int(f_s.readline().strip().split('\t')[1])
                        f.write('{}\t{}\tNear-lossless\t{}\t{}\t{}\t{}\t{}\n'.format(dec_num_blocks, dec_num_filters, near_lossless, sr_quality, bilinear_quality, feature_quality, size))

                for q in Tester.q:
                    lossy_log_dir = os.path.join(log_dir, 'webp_lossy_{}'.format(q))
                    quality_log_path = os.path.join(lossy_log_dir, 'quality.txt')
                    size_log_path = os.path.join(lossy_log_dir, 'size.txt')
                    with open(quality_log_path) as f_q, open(size_log_path) as f_s:
                        result = f_q.readline().strip().split('\t')
                        sr_quality = np.round(float(result[1]), 2)
                        bilinear_quality = np.round(float(result[2]), 2)
                        feature_quality = np.round(float(result[3]), 2)
                        size = int(f_s.readline().strip().split('\t')[1])
                        f.write('{}\t{}\tLossy\t{}\t{}\t{}\t{}\t{}\n'.format(dec_num_blocks, dec_num_filters, q, sr_quality, bilinear_quality, feature_quality, size))
