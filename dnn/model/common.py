#Reference: https://raw.githubusercontent.com/krasserm/super-resolution/master/model/common.py

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

def quantize(x, enc_min, enc_max):
    x = tf.round(255 * ((x - enc_min) / (enc_max - enc_min)))
    x = tf.clip_by_value(x, 0, 255)
    return x

def dequantize(x, enc_min, enc_max):
    x = (x_feature / 255) * (enc_max - enc_min) + enc_min
    return x

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
