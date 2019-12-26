import glob
import os

import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE


def decode_png(filepath):
    file = tf.io.read_file(filepath)
    image = tf.image.decode_png(file, channels=3)
    return image, filepath

def image_dataset(image_dir):
    images = sorted(glob.glob('{}/*.png'.format(image_dir)))
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(decode_png, num_parallel_calls=AUTOTUNE)
    return ds, len(images)

def valid_image_dataset(lr_image_dir, hr_image_dir, repeat_count=1):
    lr_ds, length = image_dataset(lr_image_dir)
    hr_ds, _ = image_dataset(hr_image_dir)

    ds = tf.data.Dataset.zip((lr_ds, hr_ds))
    ds = ds.batch(1)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
