import time
import os
import sys

import tensorflow as tf
import numpy as np

from dnn.utility import resolve, resolve_bilinear

class SingleYUVTester:
    def __init__(self, model, checkpoint_dir, log_dir, image_dir):
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                                directory=checkpoint_dir, max_to_keep=3)
        self.image_dir = image_dir
        self.log_dir = log_dir
        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def test(self, valid_dataset, save_image=False):
        sr_psnr_values = []
        bilinear_psnr_values = []

        for idx, imgs in enumerate(valid_dataset):
            self.now = time.perf_counter()

            lr_y = tf.cast(tf.expand_dims(imgs[0][0], 0), tf.float32)
            lr_u = tf.cast(tf.expand_dims(imgs[0][1], 0), tf.float32)
            lr_v = tf.cast(tf.expand_dims(imgs[0][2], 0), tf.float32)
            hr_y = tf.cast(tf.expand_dims(imgs[1][0], 0), tf.float32)
            hr_u = tf.cast(tf.expand_dims(imgs[1][1], 0), tf.float32)
            hr_v = tf.cast(tf.expand_dims(imgs[1][2], 0), tf.float32)

            if idx == 0:
                y_shape = tf.shape(hr_y)
                uv_shape = tf.shape(hr_u)
                y_width = y_shape[2]
                y_height = y_shape[1]
                uv_width = uv_shape[2]
                uv_height = uv_shape[1]

            #model, uv upsampling
            sr_y = tf.cast(resolve(self.checkpoint.model, lr_y), tf.float32)
            sr_u = tf.cast(resolve_bilinear(lr_u, y_height, y_width), tf.float32)
            sr_v = tf.cast(resolve_bilinear(lr_v, y_height, y_width), tf.float32)
            hr_u = tf.cast(resolve_bilinear(hr_u, y_height, y_width), tf.float32)
            hr_v = tf.cast(resolve_bilinear(hr_v, y_height, y_width), tf.float32)

            #yuv-rgb conversion
            sr_r = tf.clip_by_value(tf.math.round(1.164383 * (sr_y-16) + 1.596027 * (sr_v-128)), 0, 255)
            sr_g = tf.clip_by_value(tf.math.round(1.164383 * (sr_y-16) - 0.391762 * (sr_u - 128) - 0.812968 * (sr_v - 128)), 0, 255)
            sr_b = tf.clip_by_value(tf.math.round(1.164383 * (sr_y-16) + 2.017232 * (sr_u - 128)), 0, 255)
            sr_rgb = tf.cast(tf.concat([sr_r, sr_g, sr_b], axis=3), tf.uint8)

            hr_r = tf.clip_by_value(tf.math.round(1.164383 * (hr_y-16) + 1.596027 * (hr_v-128)), 0, 255)
            hr_g = tf.clip_by_value(tf.math.round(1.164383 * (hr_y-16) - 0.391762 * (hr_u - 128) - 0.812968 * (hr_v - 128)), 0, 255)
            hr_b = tf.clip_by_value(tf.math.round(1.164383 * (hr_y-16) + 2.017232 * (hr_u - 128)), 0, 255)
            hr_rgb = tf.cast(tf.concat([hr_r, hr_g, hr_b], axis=3), tf.uint8)

            #quality
            sr_psnr_value = tf.image.psnr(hr_rgb, sr_rgb, max_val=255)[0].numpy()
            sr_psnr_values.append(sr_psnr_value)

            duration = time.perf_counter() - self.now
            print(f'{idx} frame: PSNR(SR) = {sr_psnr_value:.3f} ({duration:.2f}s)')
        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}')

        #log
        quality_log_path = os.path.join(self.log_dir, 'quality.txt')
        with open(quality_log_path, 'w') as f:
            f.write('Average\t{:.2f}\t{:.2f}\n'.format(np.average(sr_psnr_values), np.average(bilinear_psnr_values)))
            for idx, psnr_values in enumerate(list(zip(sr_psnr_values, bilinear_psnr_values))):
                f.write('{}\t{:.2f}\t{:.2f}\n'.format(idx, psnr_values[0], psnr_values[1]))

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

class SingleTester:
    def __init__(self, model, checkpoint_dir, log_dir, image_dir):
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                                directory=checkpoint_dir, max_to_keep=3)
        self.image_dir = image_dir
        self.log_dir = log_dir
        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def test(self, valid_dataset, save_image=False):
        sr_psnr_values = []
        bilinear_psnr_values = []

        for idx, imgs in enumerate(valid_dataset):
            self.now = time.perf_counter()

            lr = imgs[0]
            hr = imgs[1]

            #measure height, width
            if idx == 0:
                hr_shape = tf.shape(hr)[1:3]
                height = hr_shape[0].numpy()
                width = hr_shape[1].numpy()

            #meausre sr quality
            sr = resolve(self.checkpoint.model, lr)
            sr_psnr_value = tf.image.psnr(hr, sr, max_val=255)[0].numpy()
            sr_psnr_values.append(sr_psnr_value)

            #measure bilinear quality
            bilinear = resolve_bilinear(lr, height, width)
            bilinear_psnr_value = tf.image.psnr(hr, bilinear, max_val=255)[0].numpy()
            bilinear_psnr_values.append(bilinear_psnr_value)

            #save sr images
            if save_image:
                sr_image = tf.image.encode_png(tf.squeeze(sr))
                tf.io.write_file(os.path.join(self.image_dir, '{0:04d}.png'.format(idx+1)), sr_image)

            duration = time.perf_counter() - self.now
            print(f'{idx} frame: PSNR(SR) = {sr_psnr_value:.3f}, PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')
        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

        #log
        quality_log_path = os.path.join(self.log_dir, 'quality.txt')
        with open(quality_log_path, 'w') as f:
            f.write('Average\t{:.2f}\t{:.2f}\n'.format(np.average(sr_psnr_values), np.average(bilinear_psnr_values)))
            for idx, psnr_values in enumerate(list(zip(sr_psnr_values, bilinear_psnr_values))):
                f.write('{}\t{:.2f}\t{:.2f}\n'.format(idx, psnr_values[0], psnr_values[1]))

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')
