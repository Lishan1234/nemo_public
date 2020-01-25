import glob
import os
import subprocess

import numpy as np
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

def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def resolve_bilinear(lr_batch, width, height):
    lr_batch = tf.cast(lr_batch, tf.float32)
    bilinear_batch = tf.image.resize_bilinear(lr_batch, (width, height))
    bilinear_batch = tf.clip_by_value(bilinear_batch, 0, 255)
    bilinear_batch = tf.round(bilinear_batch)
    bilinear_batch = tf.cast(bilinear_batch, tf.uint8)
    return bilinear_batch

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

def summary_raw_dataset(lr_image_dir, sr_image_dir, hr_image_dir, width, height, scale, repeat_count=1, pattern='*.raw', precision=tf.uint8):
    lr_ds, length = raw_dataset(lr_image_dir, width, height, pattern, precision)
    hr_ds, _ = raw_dataset(hr_image_dir, width * scale, height * scale, pattern, precision)
    sr_ds, _ = raw_dataset(sr_image_dir, width * scale, height * scale, pattern, precision)
    ds = tf.data.Dataset.zip((lr_ds, sr_ds, hr_ds))
    ds = ds.batch(1)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def raw_bilinear_quality(lr_raw_dir, hr_raw_dir, nhwc, scale):
    bilinear_psnr_values = []
    valid_raw_ds = valid_raw_dataset(lr_raw_dir, hr_raw_dir, nhwc[1], nhwc[2],
                                                    scale, precision=tf.float32)
    for idx, imgs in enumerate(valid_raw_ds):
        lr = imgs[0][0]
        hr = imgs[1][0]

        bilinear = resolve_bilinear(lr, nhwc[1] * scale, nhwc[2] * scale)
        bilinear = tf.cast(bilinear, tf.uint8)
        hr = tf.clip_by_value(hr, 0, 255)
        hr = tf.round(hr)
        hr = tf.cast(hr, tf.uint8)

        bilinear_psnr_value = tf.image.psnr(bilinear, hr, max_val=255)[0].numpy()
        bilinear_psnr_values.append(bilinear_psnr_value)

    return bilinear_psnr_values

def raw_sr_quality(sr_raw_dir, hr_raw_dir, nhwc, scale):
    sr_psnr_values = []
    valid_raw_ds = valid_raw_dataset(sr_raw_dir, hr_raw_dir, nhwc[1] * scale,
                                                    nhwc[2] * scale,
                                                    1, precision=tf.float32)
    for idx, imgs in enumerate(valid_raw_ds):
        sr = imgs[0][0]
        hr = imgs[1][0]

        sr = tf.clip_by_value(sr, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)
        hr = tf.clip_by_value(hr, 0, 255)
        hr = tf.round(hr)
        hr = tf.cast(hr, tf.uint8)

        sr_psnr_value = tf.image.psnr(sr, hr, max_val=255)[0].numpy()
        sr_psnr_values.append(sr_psnr_value)

    return sr_psnr_values

def raw_quality(lr_raw_dir, sr_raw_dir, hr_raw_dir, nhwc, scale):
    bilinear_psnr_values= []
    sr_psnr_values = []
    summary_raw_ds = summary_raw_dataset(lr_raw_dir, sr_raw_dir, hr_raw_dir, nhwc[1], nhwc[2],
                                                    scale, precision=tf.float32)
    for idx, imgs in enumerate(summary_raw_ds):
        lr = imgs[0][0]
        sr = imgs[1][0]
        hr = imgs[2][0]

        hr = tf.clip_by_value(hr, 0, 255)
        hr = tf.round(hr)
        hr = tf.cast(hr, tf.uint8)
        sr = tf.clip_by_value(sr, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)

        bilinear = resolve_bilinear(lr, nhwc[1] * scale, nhwc[2] * scale)
        bilinear = tf.cast(bilinear, tf.uint8)
        bilinear_psnr_value = tf.image.psnr(bilinear, hr, max_val=255)[0].numpy()
        bilinear_psnr_values.append(bilinear_psnr_value)
        sr_psnr_value = tf.image.psnr(sr, hr, max_val=255)[0].numpy()
        sr_psnr_values.append(sr_psnr_value)
        print(f'{idx} frame: PSNR(SR) = {sr_psnr_value:.3f}, PSNR(Bilinear) = {bilinear_psnr_value:3f}')
    print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

    return sr_psnr_values, bilinear_psnr_values
