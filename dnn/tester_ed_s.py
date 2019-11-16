from importlib import import_module
import time
import os
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from model.common import NormalizeConfig
from model.edsr_ed_s import EDSR_ED_S
from dataset import ImageDataset
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption

import tensorflow as tf
tf.enable_eager_execution()

class Tester:
    def __init__(self, edsr_ed_s, checkpoint_dir, log_dir, image_dir):
        if not os.path.exists(checkpoint_dir):
            raise ValueError('checkpoint_dir does not exist: {}'.format(checkpoint_dir))
        if not os.path.exists(log_dir):
            raise ValueError('log_dir does not exist: {}'.format(log_dir))
        if not os.path.exists(image_dir):
            raise ValueError('image_dir does not exist: {}'.format(image_dir))

        self.edsr_ed_s = edsr_ed_s
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.image_dir = image_dir

        model = edsr_ed_s.build_model()
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                psnr=tf.Variable(-1.0),
                                                optimizer=Adam(0),
                                                model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                        directory=checkpoint_dir, max_to_keep=3)
        self.ckpt_name = self._restore()

    def _restore(self):
        latest_ckpt = self.checkpoint_manager.latest_checkpoint
        if latest_ckpt:
            self.checkpoint.restore(latest_ckpt)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')
        else:
            raise RuntimeError('Cannot restore checkpoint')

        name = os.path.basename(latest_ckpt)
        name = os.path.splitext(name)[0]
        return name

    def encoder(self):
        model_filepath = os.path.join(self.checkpoint_dir, '{}.h5'.format(self.ckpt_name))
        if not os.path.exists(model_filepath):
            self.checkpoint.model.save_weights(model_filepath)
        encoder = self.edsr_ed_s.build_encoder()
        encoder.load_weights(model_filepath, by_name=True)
        return encoder

    def decoder(self):
        model_filepath = os.path.join(self.checkpoint_dir, '{}.h5'.format(self.ckpt_name))
        if not os.path.exists(model_filepath):
            self.checkpoint.model.save_weights(model_filepath)
        decoder = self.edsr_ed_s.build_decoder()
        decoder.load_weights(model_filepath, by_name=True)
        return decoder

    def feature(self, valid_dataset):
        feature_image_dir = os.path.join(self.image_dir, '{}_feature'.format(self.checkpoint.model.name))
        os.makedirs(feature_image_dir, exist_ok=True)

        with open(os.path.join(feature_image_dir, 'distribution.txt'), 'w') as f:
            for idx, imgs in enumerate(valid_dataset):
                self.now = time.perf_counter()
                lr = tf.cast(imgs[0], tf.float32)
                hr = imgs[1]

                lr_feature, _ = self.checkpoint.model(lr)
                lr_feature = lr_feature.numpy()

                lr_feature = lr_feature.flatten()
                result = np.percentile(lr_feature ,[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], interpolation='nearest')
                result = [str(np.round(i,2)) for i in result]
                log = '\t'.join(result)
                f.write(log)

                _ = plt.hist(lr_feature, bins='auto')
                fig_filepath = os.path.join(feature_image_dir, '{:04d}.png'.format(idx))
                plt.savefig(fig_filepath)

                duration = time.perf_counter() - self.now

                #TODO: reset plot
                #TODO: print
                #TODO: how to determine encoding_min, encoding_max?
                print(log)

    def encode(self, encoder, dataset, qnt_config, save_image=False):
        #TODO: similar to encode_decode
        pass

    def decode(self, decoder, dataset, qnt_config, save_image=False):
        #TODO: similar to encode_decode
        pass

    def encode_decode(self, encoder, decoder, dataset, qnt_config, save_image=False):
        subdir = '{}_{}'.format(encoder.name, decoder.name)
        if qnt_config: subdir += '_{}'.format(qnt_config.name)
        image_dir = os.path.join(self.image_dir, subdir)
        log_dir = os.path.join(self.log_dir, subdir)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        sr_psnr_values = []
        bilinear_psnr_values = []

        for idx, imgs in enumerate(dataset):
            self.now = time.perf_counter()

            lr = imgs[0]
            hr = imgs[1]

            #measure height, width
            if idx == 0:
                hr_shape = tf.shape(hr)[1:3]
                height = hr_shape[0].numpy()
                width = hr_shape[1].numpy()

            #meausre sr quality
            lr = tf.cast(lr, tf.float32)
            feature = encoder(lr)
            if qnt_config:
                feature = Quantize(qnt_config.enc_min, qnt_config.enc_max)
                feature = Dequantize(qnt_config.enc_min, qnt_config.enc_max)
            sr = decoder(feature)
            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)
            sr_psnr_value = tf.image.psnr(hr, sr, max_val=255)[0].numpy()
            sr_psnr_values.append(sr_psnr_value)

            #measure bilinear quality
            bilinear = resolve_bilinear(lr, height, width)
            bilinear_psnr_value = tf.image.psnr(hr, bilinear, max_val=255)[0].numpy()
            bilinear_psnr_values.append(bilinear_psnr_value)

            if save_image:
                #save sr images
                sr_image = tf.image.encode_png(tf.squeeze(sr))
                tf.io.write_file(os.path.join(image_dir, '{0:04d}.png'.format(idx)), sr_image)

            duration = time.perf_counter() - self.now
            print(f'PSNR(SR) = {sr_psnr_value:.3f}, PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')
        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

        log_filepath = os.path.join(log_dir, 'quality.log')
        with open(log_filepath, 'w') as f:
            for psnr_values in list(zip(sr_psnr_values, bilinear_psnr_values)):
                log = '{:.2f}\t{:.2f}\n'.format(psnr_values[0], psnr_values[1])
                f.write(log)

    def finetune_decoder(self, train_dataset, valid_dataset, save_image=False):
        pass

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

    #log
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--custom_tag', type=str, default=None)

    args = parser.parse_args()

    #0. setting
    video_dir = os.path.join(args.dataset_dir, 'video')
    image_dir = os.path.join(args.dataset_dir, 'image')
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint')
    log_dir = os.path.join(args.dataset_dir, 'log')
    if not os.path.exists(video_dir):
        raise ValueError('directory does not exist: {}'.format(video_dir))
    if not os.path.exists(image_dir):
        raise ValueError('directory does not exist: {}'.format(image_dir))
    if not os.path.exists(checkpoint_dir):
        raise ValueError('directory does not exist: {}'.format(checkpoint_dir))
    if not os.path.exists(log_dir):
        raise ValueError('directory does not exist: {}'.format(log_dir))

    #1. creat datasets
    video_metadata = VideoMetadata(args.video_format, args.start_time, args.duration)
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
    dataset = ImageDataset(video_dir,
                                image_dir,
                                video_metadata,
                                ffmpeg_option,
                                args.ffmpeg_path)

    with tf.device('cpu:0'):
        train_ds, valid_ds, train_dir, valid_dir, rgb_mean, scale = dataset.dataset(args.input_resolution,
                                                args.target_resolution,
                                                args.batch_size,
                                                args.patch_size,
                                                args.load_on_memory)
    #2. create a DNN
    if args.enable_normalization:
        normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
    else:
        normalize_config = None
    edsr_ed_s = EDSR_ED_S(args.enc_num_blocks, args.enc_num_filters, \
                           args.dec_num_blocks, args.dec_num_filters, \
                            scale, None)

    #3. create a tester
    dataset_tag = '{}.{}'.format(video_metadata.summary(args.input_resolution, True), ffmpeg_option.summary())
    model_tag = '{}'.format(edsr_ed_s.name)
    checkpoint_dir = os.path.join(checkpoint_dir, dataset_tag, model_tag)
    log_dir = os.path.join(log_dir, dataset_tag)
    image_dir = os.path.join(image_dir, dataset_tag)
    tester = Tester(edsr_ed_s, checkpoint_dir, log_dir, image_dir)

    #4. get encoder, decoder
    encoder = tester.encoder()
    decoder = tester.decoder()

    #5. evaluate encoder-decoder
    tester.encode_decode(encoder, decoder, valid_ds, None, True)
