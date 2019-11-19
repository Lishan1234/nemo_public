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

def resolve_bilinear(lr_batch, width, height):
    lr_batch = tf.cast(lr_batch, tf.float32)
    bilinear_batch = tf.image.resize_bilinear(lr_batch, (width, height))
    bilinear_batch = tf.clip_by_value(bilinear_batch, 0, 255)
    bilinear_batch = tf.round(bilinear_batch)
    bilinear_batch = tf.cast(bilinear_batch, tf.uint8)
    return bilinear_batch

def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = tf.image.psnr(hr, sr, max_val=255)[0]
        psnr_values.append(psnr_value)
    return psnr_values, tf.reduce_mean(psnr_values)

class FFmpegOption():
    def __init__(self, filter_type, filter_fps, upsample):
        if filter_type not in ['key', 'uniform', 'none']:
            raise ValueError('filter type is not valid: {}'.format(filter_type))
        if filter_type is 'uniform' and filter_fps is None:
            raise ValueError('filter fps is not set: {}'.format(filter_fps))
        if upsample not in ['bilinear']:
            raise ValueError('upsample is not valid: {}'.format(upsample))

        self.filter_type = filter_type
        self.filter_fps = filter_fps
        self.upsample = upsample

    def summary(self, video_name):
        if self.filter_type == 'key':
            return '{}.key'.format(video_name)
        elif self.filter_type == 'uniform':
            return '{}.uniform_{:.2f}'.format(video_name, self.filter_fps)
        elif self.filter_type == 'none':
            return video_name

    def filter(self):
        if self.filter_type == 'key':
            return '-vf "select=eq(pict_type\,I)" -vsync vfr'
        elif self.filter_type == 'uniform':
            return '-vf fps={}'.format(self.filter_fps)
        elif self.filter_type == 'none':
            return ''

    def filter_rescale(self, width, height):
        if self.filter_type == 'key':
            return '-vf "select=eq(pict_type\,I)",scale={}:{} -vsync vfr -sws_flags {}'.format(width, height, self.upsample)
        elif self.filter_type == 'uniform':
            return '-vf fps={},scale={}:{} -sws_flags {}'.format(self.filter_fps, width, height, self.upsample)
        elif self.filter_type == 'none':
            return '-vf scale={}:{} -sws_flags {}'.format(width, height, self.upsample)

#TODO: filter with a cache profile
"""
1. load a cache profile
2. if a visible frame is set as a cnhor point
    add to target frame list
3. use ffmpeg to extract/save filtered frames with index from 2.
link: https://stackoverflow.com/questions/38253406/extract-list-of-specific-frames-using-ffmpeg
"""

# ---------------------------------------
# Video
# ---------------------------------------

class VideoMetadata():
    def __init__(self, video_format, start_time, duration):
        self.video_format = video_format
        self.start_time = start_time
        self.duration = duration

    #TODO: add bitrate and vidoe_format
    def summary(self, resolution, is_encoded):
        name = '{}p'.format(resolution)
        if self.start_time is not None:
            name += '_s{}'.format(self.start_time)
        if self.duration is not None:
            name += '_d{}'.format(self.duration)
        if is_encoded: name += '_encoded'
        name += '.{}'.format(self.video_format)
        return name

# ---------------------------------------
# Entropy
# ---------------------------------------

#TODO
def measure_entropy(model, dataset, quantization_config):
    lr_entropy = []
    feature_entropy = []

    if not os.path.exists(log_path):
        with open(ent_log_path, 'w') as f:
            for idx, imgs in enumerate(dataset):
                now = time.perf_counter()
                lr = tf.cast(imgs[0], tf.float32)
                feature = model(lr)

                lr = lr.numpy()
                feature = feature.numpy()

                lr_gray = rgb2gray(lr)
                feature_gray= rgb2gray(feature)

                lr_entropy_value = shannon_entropy(lr_gray)
                feature_entropy_value = shannon_entropy(feature_gray)
                lr_entropy.append(lr_entropy_value)
                feature_entorpy.append(feature_entropy_value)

                f.write('{:.2f}\t{:.2f}\n'.format(lr_entropy_value, feature_entropy_value))

                duration = time.perf_counter() - self.now
                print('lr_entropy={:.2f} feature_entropy={:.2f} ({:.2f}s)'.format(lr_entropy, feature_entropy, duration))

    return lr_entropy, feature_entropy
