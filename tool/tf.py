import glob
import os
import subprocess

import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

def get_tensorflow_dir():
    tensorflow_dir = None

    cmd = 'pip show tensorflow'
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = proc.stdout.readlines()
    for line in lines:
        line = line.decode().rstrip('\r\n')
        if 'Version' in line:
            tensorflow_ver = line.split(' ')[1]
            if not tensorflow_ver.startswith('1.'):
                raise RuntimeError('Tensorflow verion is wrong: {}'.format(tensorflow_ver))
        if 'Location' in line:
            tensorflow_dir = line.split(' ')[1]
            tensorflow_dir = os.path.join(tensorflow_dir, 'tensorflow')
    if tensorflow_dir is None:
        raise RuntimeError('Tensorflow is not installed')

    return tensorflow_dir

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

def decode_raw(filepath, width, height, precision):
    file = tf.io.read_file(filepath)
    image = tf.decode_raw(file, precision)
    image = tf.reshape(image, [width, height, 3])
    return image, filepath

def raw_dataset(image_dir, width, height, pattern, precision):
    images = sorted(glob.glob('{}/{}'.format(image_dir, pattern)))
    #images = sorted(glob.glob('{}/[0-9][0-9][0-9][0-9].raw'.format(image_dir)))
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(lambda x: decode_raw(x, width, height, precision), num_parallel_calls=AUTOTUNE)
    return ds, len(images)

def single_raw_dataset(image_dir, width, height, repeat_count=1, pattern='*.raw', precision=tf.uint8):
    ds, length = raw_dataset(image_dir, width, height, pattern, precision)
    ds = ds
    ds = ds.batch(1)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def valid_raw_dataset(lr_image_dir, hr_image_dir, width, height, scale, repeat_count=1, pattern='*.raw', precision=tf.uint8):
    lr_ds, length = raw_dataset(lr_image_dir, width, height, pattern, precision)
    hr_ds, _ = raw_dataset(hr_image_dir, width * scale, height * scale, pattern, precision)
    ds = tf.data.Dataset.zip((lr_ds, hr_ds))
    ds = ds.batch(1)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
