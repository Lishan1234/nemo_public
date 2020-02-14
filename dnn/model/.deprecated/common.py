#Reference: https://raw.githubusercontent.com/krasserm/super-resolution/master/model/common.py
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import single_image_dataset

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0

def denormalize_01(x):
    return x * 255.0

def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1

def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5

class NormalizeConfig():
    def __init__(self, normalize_layer, denormalize_layer, rgb_mean=None):
        if not normalize_layer in globals():
            raise ValueError('normalizae layer is not supported: {}'.format(normalize_layer))
        if not denormalize_layer in globals():
            raise ValueError('denormalizae layer is not supported: {}'.format(denormalize_layer))
        #TODO: check the number of function arguemtns (http://xahlee.info/python/python_get_number_of_args.html)

        self.normalize_layer = globals()[normalize_layer]
        self.denormalize_layer = globals()[denormalize_layer]
        self.rgb_mean = rgb_mean

    def normalize(self, x):
        if self.rgb_mean is not None:
            return self.normalize_layer(x, self.rgb_mean)
        else:
            return self.normalize_layer(x)

    def denormalize(self, x):
        if self.rgb_mean is not None:
            return self.denormalize_layer(x, self.rgb_mean)
        else:
            return self.denormalize_layer(x)

# ---------------------------------------
#  Quantization
# ---------------------------------------

class LinearQuantizer():
    log_name = 'distribution.txt'
    percentiles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, \
                    95.0, 98.0, 99.0, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9, 100.0]

    #percentiles = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, \
    #                80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0]

    def __init__(self, min_percentile, max_percentile):
        assert(min_percentile in self.percentiles)
        assert(max_percentile in self.percentiles)

        self.min_percentile = min_percentile
        self.min_idx = self.percentiles.index(min_percentile)
        self.max_percentile = max_percentile
        self.max_idx = self.percentiles.index(max_percentile)

        self.min_value = None
        self.max_value = None

    @property
    def name(self):
        return 'linear_{:.2f},{:.2f}'.format(self.min_percentile, self.max_percentile)

    def load(self, log_dir, min_select=np.min, max_select=np.max):
        log_path = os.path.join(log_dir, self.log_name)
        assert(os.path.exists(log_path))
        features = []

        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                feature = [float(i) for i in line]
                features.append(feature)
        features_arr = np.array(features)

        self.min_value = min_select(features_arr[:,self.min_idx])
        self.max_value = max_select(features_arr[:,self.max_idx])

        print('load: min_value {:.2f}, max_value {:.2f}'.format(self.min_value, self.max_value))

    def profile(self, model, image_dir, log_dir):
        log_path = os.path.join(log_dir, self.log_name)
        dataset = single_image_dataset(image_dir)

        with open(log_path, 'w') as f:
            for idx, img in enumerate(dataset):
                now = time.perf_counter()
                lr = tf.cast(img, tf.float32)
                lr_feature = model(lr)
                lr_feature = lr_feature.numpy()
                lr_feature = lr_feature.flatten()

                result = np.percentile(lr_feature , self.percentiles, interpolation='nearest')
                result = [np.round(i,2) for i in result]
                log = '\t'.join([str(i) for i in result])
                log += '\n'
                f.write(log)

                duration = time.perf_counter() - now
                print('0%-percentile={:.2f} 100%-percentile={:.2f} (duration: {:.2f}s)'.format(result[0], result[-1], duration))

    def plot(self, model, image_dir, log_dir):
        dataset = single_image_dataset(image_dir)

        for idx, img in enumerate(dataset):
            now = time.perf_counter()
            lr = tf.cast(img, tf.float32)
            lr_feature = model(lr)
            lr_feature = lr_feature.numpy()
            lr_feature = lr_feature.flatten()

            _ = plt.hist(lr_feature, bins='auto')
            fig_filepath = os.path.join(log_dir, '{:04d}.png'.format(idx))
            plt.savefig(fig_filepath)
            plt.clf()

            duration = time.perf_counter() - now
            print('(duration: {:.2f}s)'.format(duration))

    def quantize(self, x):
        assert(self.min_value is not None)
        assert(self.max_value is not None)

        x = tf.round(255 * ((x - self.min_value) / (self.max_value - self.min_value)))
        x = tf.clip_by_value(x, 0, 255)
        return x

    def dequantize(self, x):
        assert(self.min_value is not None)
        assert(self.max_value is not None)

        x = (x / 255) * (self.max_value - self.min_value) + self.min_value
        return x

# ---------------------------------------
#  Metrics
# ---------------------------------------

def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def test():
    print(globals())

if __name__ == '__main__':
    print(globals())
