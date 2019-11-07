from importlib import import_module
import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from utility import resolve, resolve_bilinear

class Tester:
    def __init__(self, model, checkpoint_dir, log_dir, image_dir):
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                psnr=tf.Variable(-1.0),
                                                optimizer=Adam(0),
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

                sr_image_dir = os.path.join(self.image_dir, '{}_{}x{}'.format(self.checkpoint.model.name, height, width))
                bilinear_image_dir = os.path.join(self.image_dir, 'bilinear_{}x{}'.format(height, width))
                os.makedirs(sr_image_dir, exist_ok=True)
                os.makedirs(bilinear_image_dir, exist_ok=True)

            #meausre sr quality
            sr = resolve(self.checkpoint.model, lr)
            sr_psnr_value = tf.image.psnr(hr, sr, max_val=255)[0].numpy()
            sr_psnr_values.append(sr_psnr_value)

            #measure bilinear quality
            bilinear = resolve_bilinear(lr, height, width)
            bilinear_psnr_value = tf.image.psnr(hr, bilinear, max_val=255)[0].numpy()
            bilinear_psnr_values.append(bilinear_psnr_value)

            if save_image:
                #save sr images
                sr_image = tf.image.encode_png(tf.squeeze(sr))
                tf.io.write_file(os.path.join(sr_image_dir, '{0:04d}.png'.format(idx)), sr_image)

                #save bilinear images
                bilinear_image = tf.image.encode_png(tf.squeeze(bilinear))
                tf.io.write_file(os.path.join(bilinear_image_dir, '{0:04d}.png'.format(idx)), bilinear_image)

            duration = time.perf_counter() - self.now
            print(f'PSNR(SR) = {sr_psnr_value:.3f}, PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')
        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

        sr_log = os.path.join(self.log_dir, 'quality_{}_{}x{}.log'.format(self.checkpoint.model.name, height, width))
        bilinear_log = os.path.join(self.log_dir, 'quality_bilinear_{}x{}.log'.format(height, width))

        with open(sr_log, 'w') as sr_f, open(bilinear_log, 'w') as bilinear_f:
            for sr_psnr_value in sr_psnr_values:
                sr_f.write('{0:.2f}\n'.format(sr_psnr_value))

            for bilinear_psnr_value in bilinear_psnr_values:
                bilinear_f.write('{0:.2f}\n'.format(bilinear_psnr_value))

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

if __name__ == '__main__':
    pass
