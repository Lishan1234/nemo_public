#Reference: https://raw.githubusercontent.com/krasserm/super-resolution/master/model/common.py
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

class QuantizeConfig():
    log_name = 'feature_distribution.txt'

    def __init__(self, policy):
        assert policy in ['min_max_0']

        self.policy = policy
        self.enc_min = None
        self.enc_max = None

    def load(self, log_path):
        features = []
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                feature = [float(i) for i in line]
                features.append(feature)
        features_arr = np.array(features)

        if self.policy == 'min_max_0':
            self.enc_min = np.min(features_arr[:,0])
            self.enc_max = np.max(features_arr[:,-1])

    def set(self, model, dataset, log_dir, image_dir):
        log_path = os.path.join(log_dir, self.log_name)

        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                for idx, imgs in enumerate(dataset):
                    now = time.perf_counter()
                    lr = tf.cast(imgs, tf.float32)
                    lr_feature = model(lr)
                    lr_feature = lr_feature.numpy()
                    lr_feature = lr_feature.flatten()

                    result = np.percentile(lr_feature ,[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], interpolation='nearest')
                    result = [np.round(i,2) for i in result]
                    log = '\t'.join([str(i) for i in result])
                    log += '\n'
                    f.write(log)

                    _ = plt.hist(lr_feature, bins='auto')
                    fig_filepath = os.path.join(image_dir, '{:04d}.png'.format(idx))
                    plt.savefig(fig_filepath)
                    plt.clf()

                    duration = time.perf_counter() - now
                    print('0%-percentile={:.2f} 100%-percentile={:.2f} ({:.2f}s)'.format(result[0], result[-1], duration))

        self.load(log_path)

    @property
    def name(self):
        name = 'min{}_max{}'.format(round(self.enc_min), round(self.enc_max))
        return name

def quantize(x, enc_min, enc_max):
    x = tf.round(255 * ((x - enc_min) / (enc_max - enc_min)))
    x = tf.clip_by_value(x, 0, 255)
    return x

def dequantize(x, enc_min, enc_max):
    x = (x / 255) * (enc_max - enc_min) + enc_min
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
