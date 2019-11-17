from importlib import import_module
import time
import os
import sys
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from model.common import NormalizeConfig, QuantizeConfig, quantize, dequantize
from model.edsr_ed_s import EDSR_ED_S
from dataset import ImageDataset, image_valid_dataset
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption

import tensorflow as tf
tf.enable_eager_execution()

class Tester:
    quantization_subdir = 'feature'
    quantization_log_name = 'distribution.txt'
    encode_subdir = 'encode'
    decode_subdir = 'decode'

    def __init__(self, edsr_ed_s, checkpoint_dir, log_dir, lr_image_dir, hr_image_dir):
        self.edsr_ed_s = edsr_ed_s
        self.lr_image_dir = lr_image_dir
        self.hr_image_dir = hr_image_dir
        self.checkpoint_dir = os.path.join(checkpoint_dir, edsr_ed_s.name)

        self.quantization_image_dir = os.path.join(self.lr_image_dir, edsr_ed_s.name, self.quantization_subdir)
        self.quantization_log_dir = os.path.join(log_dir, edsr_ed_s.name, self.quantization_subdir)
        self.encode_image_dir = os.path.join(self.lr_image_dir, edsr_ed_s.name, self.encode_subdir)
        self.decode_image_dir = os.path.join(self.lr_image_dir, edsr_ed_s.name, self.decode_subdir)
        self.decode_log_dir = os.path.join(log_dir, edsr_ed_s.name, self.decode_subdir)
        os.makedirs(self.quantization_image_dir, exist_ok=True)
        os.makedirs(self.quantization_log_dir, exist_ok=True)
        os.makedirs(self.encode_image_dir, exist_ok=True)
        os.makedirs(self.decode_image_dir, exist_ok=True)
        os.makedirs(self.decode_log_dir, exist_ok=True)

        self.encoder = None
        self.decoder = None
        self.quantization_config = None

    def _convert_to_h5(self, checkpoint_dir):
        model = self.edsr_ed_s.build_model()
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                            psnr=tf.Variable(-1.0),
                                            optimizer=Adam(0),
                                            model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                        directory=checkpoint_dir, max_to_keep=3)

        print(checkpoint_dir)
        checkpoint_path = checkpoint_manager.latest_checkpoint
        assert(checkpoint_path is not None)
        h5_path = '{}.h5'.format(os.path.splitext(checkpoint_path)[0])

        if not os.path.exists(h5_path):
            checkpoint.restore(checkpoint_path)
            checkpoint.model.save_weights(h5_path)

        return h5_path

    def load_encoder(self, checkpoint_dir):
        if checkpoint_dir is None: checkpoint_dir = self.checkpoint_dir
        h5_path = self._convert_to_h5(checkpoint_dir)
        encoder = self.edsr_ed_s.build_encoder()
        encoder.load_weights(h5_path, by_name=True)
        self.encoder = encoder

    def load_decoder(self, checkpoint_dir):
        if checkpoint_dir is None: checkpoint_dir = self.checkpoint_dir
        h5_path = self._convert_to_h5(checkpoint_dir)
        decoder = self.edsr_ed_s.build_decoder()
        decoder.load_weights(h5_path, by_name=True)
        self.decoder = decoder

    def set_quantization_config(self, policy):
        assert(policy in ['min_max_0'])

        qnt_log_path = os.path.join(self.quantization_log_dir, self.quantization_log_name)

        if not os.path.exists(qnt_log_path):
            dataset = image_valid_dataset(self.lr_image_dir, hr_image_dir)
            with open(qnt_log_path, 'w') as f:
                for idx, imgs in enumerate(dataset):
                    self.now = time.perf_counter()
                    lr = tf.cast(imgs[0], tf.float32)
                    lr_feature = self.encoder(lr)
                    lr_feature = lr_feature.numpy()
                    lr_feature = lr_feature.flatten()

                    result = np.percentile(lr_feature ,[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], interpolation='nearest')
                    result = [np.round(i,2) for i in result]
                    log = '\t'.join([str(i) for i in result])
                    log += '\n'
                    f.write(log)

                    _ = plt.hist(lr_feature, bins='auto')
                    fig_filepath = os.path.join(self.quantization_image_dir, '{:04d}.png'.format(idx))
                    plt.savefig(fig_filepath)
                    plt.clf()

                    duration = time.perf_counter() - self.now
                    print('0%-percentile={:.2f} 100%-percentile={:.2f} ({:.2f}s)'.format(result[0], result[-1], duration))

        features = []
        with open(qnt_log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                feature = [float(i) for i in line]
                features.append(feature)
        features_arr = np.array(features)

        if policy == 'min_max_0':
            enc_min = np.min(features_arr[:,0])
            enc_max = np.max(features_arr[:,-1])
            qnt_config = QuantizeConfig(enc_min, enc_max)

        self.quantization_config = qnt_config

    def encode(self, dataset, save_image):
        assert(self.encoder is not None)

        if dataset is None: dataset = image_valid_dataset(self.lr_image_dir, self.hr_image_dir)

        for idx, imgs in enumerate(dataset):
            self.now = time.perf_counter()
            lr = imgs[0]

            #encode images
            lr = tf.cast(lr, tf.float32)
            feature = self.encoder(lr)
            if self.quantization_config:
                feature = quantize(feature, self.quantization_config.enc_min, self.quantization_config.enc_max)
            feature = tf.cast(feature, tf.uint8)

            if save_image:
                #save feature images
                feature_image = tf.image.encode_png(tf.squeeze(feature))
                tf.io.write_file(os.path.join(self.encode_image_dir, '{0:04d}.png'.format(idx)), feature_image)

            duration = time.perf_counter() - self.now
            print(f'({duration:.2f}s)')

    #Note: need refactoring when a network outputs LR
    def decode(self, dataset, save_image):
        assert(self.decoder is not None)

        if dataset is None: dataset = image_valid_dataset(self.lr_image_dir, self.hr_image_dir, self.encode_image_dir)
        sr_psnr_values = []
        bilinear_psnr_values = []

        for idx, imgs in enumerate(dataset):
            self.now = time.perf_counter()

            lr = imgs[0]
            hr = imgs[1]
            feature = imgs[2]

            #measure height, width
            if idx == 0:
                hr_shape = tf.shape(hr)[1:3]
                height = hr_shape[0].numpy()
                width = hr_shape[1].numpy()

            #meausre sr quality
            feature = tf.cast(feature, tf.float32)
            if self.quantization_config:
                feature = dequantize(feature, self.quantization_config.enc_min, self.quantization_config.enc_max)
            sr = self.decoder(feature)
            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)
            sr_psnr_value = tf.image.psnr(hr, sr, max_val=255)[0].numpy()
            sr_psnr_values.append(sr_psnr_value)

            #measure bilinear quality
            lr = tf.cast(lr, tf.float32)
            bilinear = resolve_bilinear(lr, height, width)
            bilinear_psnr_value = tf.image.psnr(hr, bilinear, max_val=255)[0].numpy()
            bilinear_psnr_values.append(bilinear_psnr_value)

            if save_image:
                #save sr images
                sr_image = tf.image.encode_png(tf.squeeze(sr))
                tf.io.write_file(os.path.join(self.decode_image_dir, '{0:04d}.png'.format(idx)), sr_image)

            duration = time.perf_counter() - self.now
            print(f'PSNR(SR) = {sr_psnr_value:.3f}, PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')
        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

        log_filepath = os.path.join(self.decode_log_dir, 'quality.log')
        with open(log_filepath, 'w') as f:
            for psnr_values in list(zip(sr_psnr_values, bilinear_psnr_values)):
                log = '{:.2f}\t{:.2f}\n'.format(psnr_values[0], psnr_values[1])
                f.write(log)

if __name__ == '__main__':
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
    parser.add_argument('--enable_normalization', action='store_true')
    parser.add_argument('--enable_quantization', action='store_true')
    parser.add_argument('--quantization_policy', type=str, default='min_max_0')

    #log
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--custom_tag', type=str, default=None)

    #task
    parser.add_argument('--task', type=str, required=True)

    args = parser.parse_args()

    #setting
    image_rootdir = os.path.join(args.dataset_dir, 'image')
    checkpoint_rootdir = os.path.join(args.dataset_dir, 'checkpoint')
    log_rootdir = os.path.join(args.dataset_dir, 'log')
    video_metadata = VideoMetadata(args.video_format, args.start_time, args.duration)
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)

    #dnn
    scale = math.floor(args.target_resolution / args.input_resolution)
    if args.enable_normalization:
        #TODO: rgb mean
        #normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
        pass
    else:
        normalize_config = None
    edsr_ed_s = EDSR_ED_S(args.enc_num_blocks, args.enc_num_filters, \
                           args.dec_num_blocks, args.dec_num_filters, \
                            scale, normalize_config)

    #tester
    lr_video_name = video_metadata.summary(args.input_resolution, True)
    hr_video_name = video_metadata.summary(args.target_resolution, False)
    lr_image_name = '{}.{}'.format(lr_video_name, ffmpeg_option.summary())
    hr_image_name = '{}.{}'.format(hr_video_name, ffmpeg_option.summary())
    lr_image_dir = os.path.join(image_rootdir, lr_image_name)
    hr_image_dir = os.path.join(image_rootdir, hr_image_name)
    checkpoint_dir = os.path.join(checkpoint_rootdir, lr_image_name)
    log_dir = os.path.join(log_rootdir, lr_image_name)

    tester = Tester(edsr_ed_s, checkpoint_dir, log_dir, lr_image_dir, hr_image_dir)
    tester.load_encoder(None)
    tester.load_decoder(None)
    if args.enable_quantization:
        tester.set_quantization_config(args.quantization_policy)
    else:
        qnt_config = None

    #task
    assert(args.task in ['encode', 'encode_video', 'decode', 'decode_video', 'all', 'all_video'])
    if args.task == 'encode':
        tester.encode(None, True)
    elif args.task == 'encode_video':
        #TODO
        pass
    elif args.task == 'decode':
        tester.decode(None, args.save_image)
    elif args.task == 'decode_video':
        #TODO
        pass
    elif args.task == 'all':
        tester.encode(None, True)
        tester.decode(None, args.save_image)
        pass
    elif args.task == 'all_video':
        #TODO
        pass
