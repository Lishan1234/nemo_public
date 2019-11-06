#Reference: https://github.com/krasserm/super-resolution/blob/master/model/common.py
import time

import tensorflow as tf

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
        psnr_value = tf.image.psnr(hr, sr, max_val=255)[0]
        psnr_values.append(psnr_value)
    return psnr_values, tf.reduce_mean(psnr_values)

class FFmpegOption():
    def __init__(self, filter_type, filter_fps, upsample):
        if filter_type not in ['key', 'uniform']:
            raise ValueError('filter type is not valid: {}'.format(filter_type))
        if filter_type is 'uniform' and filter_fps is None:
            raise ValueError('filter fps is not set: {}'.format(filter_fps))
        if upsample not in ['bilinear']:
            raise ValueError('upsample is not valid: {}'.format(upsample))

        self.filter_type = filter_type
        self.filter_fps = filter_fps
        self.upsample = upsample

    def summary(self):
        if self.filter_type == 'key':
            return 'key'
        elif self.filter_type == 'uniform':
            return 'uniform_{0:.2f}'.format(self.filter_fps)

    def filter(self):
        if self.filter_type == 'key':
            return '-vf "select=eq(pict_type\,I)" -vsync vfr'
        elif self.filter_type == 'uniform':
            return '-vf fps={}'.format(self.filter_fps)

    def filter_rescale(self, width, height):
        if self.filter_type == 'key':
            return '-vf "select=eq(pict_type\,I)",scale={}:{} -vsync vfr -sws_flags {}'.format(width, height, self.upsample)
        elif self.filter_type == 'uniform':
            return '-vf fps={},scale={}:{} -sws_flags {}'.format(self.filter_fps, width, height, self.upsample)

#TODO: filter with a cache profile
"""
1. load a cache profile
2. if a visible frame is set as a cnhor point
    add to target frame list
3. use ffmpeg to extract/save filtered frames with index from 2.
link: https://stackoverflow.com/questions/38253406/extract-list-of-specific-frames-using-ffmpeg
"""

class VideoMetadata():
    def __init__(self, video_format, start_time, duration):
        self.video_format = video_format
        self.start_time = start_time
        self.duration = duration

    def summary(self, resolution, is_encoded):
        name = '{}p'.format(resolution)
        if self.start_time is not None:
            name += '_s{}'.format(self.start_time)
        if self.duration is not None:
            name += '_d{}'.format(self.duration)
        if is_encoded: name += '_encoded'
        name += '.{}'.format(self.video_format)
        return name
