import time
import os

import tensorflow as tf
import numpy as np

from dnn.utility import resolve, resolve_bilinear

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
