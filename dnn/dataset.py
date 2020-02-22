import os, glob, sys, time
import logging
import math
import shlex
import subprocess
import json
import struct
import re

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.data.experimental import AUTOTUNE

from tool.video import FFmpegOption, VideoMetadata, profile_video

#TODO: check memory usage for multiple resolutions (e.g., share target resolution frames)
#TODO: check available memory to decide load on memory or on storage

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def setup_images(video_path, image_dir, ffmpeg_path, ffmpeg_option):
    os.makedirs(image_dir, exist_ok=True)
    video_name = os.path.basename(video_path)
    cmd = '{} -i {} {} {}/%04d.png'.format(ffmpeg_path, video_path, ffmpeg_option, image_dir)
    os.system(cmd)

def setup_yuv_images(vpxdec_file, content_dir, video_file, filter_fps):
    image_dir = os.path.join(content_dir, 'image', os.path.basename(video_file), 'libvpx')
    os.makedirs(image_dir, exist_ok=True)
    #fps
    video_profile = profile_video(video_file)
    fps = video_profile['frame_rate']
    filter_interval = math.floor(fps / filter_fps)

    #decode
    command = '{} --codec=vp9 --progress --summary --noblit --threads=1 --frame-buffers=50  \
    --content-dir={} --input-video={} --filter-interval={} --postfix=libvpx --save-yuvframe'.format(vpxdec_file,
                    content_dir, os.path.basename(video_file), filter_interval)
    subprocess.check_call(shlex.split(command), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def random_crop(lr_image, hr_image, lr_crop_size, scale):
    lr_image_shape = tf.shape(lr_image)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_image_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_image_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_crop_size = lr_crop_size * scale
    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_image_cropped = lr_image[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_image_cropped = hr_image[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_image_cropped, hr_image_cropped

def random_crop_feature(lr_image, feature_image, hr_image, lr_crop_size, scale):
    lr_image_shape = tf.shape(lr_image)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_image_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_image_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_crop_size = lr_crop_size * scale
    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_image_cropped = lr_image[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    feature_image_cropped = feature_image[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_image_cropped = hr_image[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_image_cropped, feature_image_cropped, hr_image_cropped

def image_dataset(image_dir, exp):
    m = re.compile(exp)
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if m.search(f)])
    #images = sorted(glob.glob('{}/*.png'.format(image_dir)))
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(tf.io.read_file)
    ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
    return ds, len(images)

def single_image_dataset(image_dir, exp='.png'):
    ds, _ = image_dataset(image_dir, exp)
    ds = ds.batch(1)
    ds = ds.repeat(1)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def train_image_dataset(lr_dir, hr_dir, batch_size, patch_size, scale, load_on_memory, repeat_count=None, exp='.png'):
    lr_ds, num_images = image_dataset(lr_dir, exp)
    hr_ds, num_images = image_dataset(hr_dir, exp)
    print('number of images: {}'.format(num_images))

    ds = tf.data.Dataset.zip((lr_ds, hr_ds))
    if load_on_memory: ds = ds.cache()
    ds = ds.shuffle(buffer_size=num_images)
    ds = ds.map(lambda lr, hr: random_crop(lr, hr, patch_size, scale), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def valid_image_dataset(lr_dir, hr_dir, repeat_count=1, exp='.png'):
    lr_ds, _ = image_dataset(lr_dir, exp)
    hr_ds, _ = image_dataset(hr_dir, exp)

    ds = tf.data.Dataset.zip((lr_ds, hr_ds))
    ds = ds.batch(1)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def train_feature_dataset(lr_dir, feature_dir, hr_dir, batch_size, patch_size, scale, load_on_memory, repeat_count=None):
    lr_ds, num_images = image_dataset(lr_dir)
    feature_ds, num_images = image_dataset(feature_dir)
    hr_ds, num_images = image_dataset(hr_dir)
    print('number of images: {}'.format(num_images))

    ds = tf.data.Dataset.zip((lr_ds, feature_ds, hr_ds))
    if load_on_memory: ds = ds.cache()
    ds = ds.shuffle(buffer_size=num_images)
    ds = ds.map(lambda lr, feature, hr: random_crop_feature(lr, feature, hr, patch_size, scale), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def valid_feature_dataset(lr_dir, feature_dir, hr_dir):
    lr_ds, _ = image_dataset(lr_dir)
    feature_ds, _ = image_dataset(feature_dir)
    hr_ds, _ = image_dataset(hr_dir)

    ds = tf.data.Dataset.zip((lr_ds, feature_ds, hr_ds))
    ds = ds.batch(1)
    ds = ds.repeat(1)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def decode_raw(filepath, width, height, channel, precision):
    file = tf.io.read_file(filepath)
    image = tf.decode_raw(file, precision)
    image = tf.reshape(image, [height, width, channel])
    #return image, filepath
    return image

def decode_raw_with_name(filepath, width, height, channel, precision):
    file = tf.io.read_file(filepath)
    image = tf.decode_raw(file, precision)
    image = tf.reshape(image, [height, width, channel])
    return image, filepath

def raw_dataset(image_dir, width, height, channel, exp, precision):
    m = re.compile(exp)
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if m.search(f)])
    #images = sorted(glob.glob('{}/{}'.format(image_dir, pattern)))
    #images = sorted(glob.glob('{}/[0-9][0-9][0-9][0-9].raw'.format(image_dir)))
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(lambda x: decode_raw(x, width, height, channel, precision), num_parallel_calls=AUTOTUNE)
    return ds, len(images)

def single_raw_dataset(image_dir, width, height, channel, exp, repeat_count=1, precision=tf.uint8):
    ds, length = raw_dataset(image_dir, width, height, channel, exp, precision)
    ds = ds.batch(1)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def single_raw_dataset_with_name(image_dir, width, height, channel, exp, repeat_count=1, precision=tf.uint8):
    m = re.compile(exp)
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if m.search(f)])
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(lambda x: decode_raw_with_name(x, width, height, channel, precision), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(1)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def train_raw_dataset(lr_image_dir, hr_image_dir, width, height, channel, scale, batch_size, patch_size, load_on_memory, exp, repeat_count=None, precision=tf.uint8):
    lr_ds, length = raw_dataset(lr_image_dir, width, height, channel, exp, precision)
    hr_ds, _ = raw_dataset(hr_image_dir, width * scale, height * scale, channel, exp, precision)
    ds = tf.data.Dataset.zip((lr_ds, hr_ds))
    if load_on_memory: ds = ds.cache()
    ds = ds.shuffle(buffer_size=length)
    ds = ds.map(lambda lr, hr: random_crop(lr, hr, patch_size, scale), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def valid_raw_dataset(lr_image_dir, hr_image_dir, width, height, channel, scale, exp, repeat_count=1, precision=tf.uint8):
    lr_ds, length = raw_dataset(lr_image_dir, width, height, channel, exp, precision)
    hr_ds, _ = raw_dataset(hr_image_dir, width * scale, height * scale, channel, exp, precision)
    ds = tf.data.Dataset.zip((lr_ds, hr_ds))
    ds = ds.batch(1)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def summary_raw_dataset(lr_image_dir, sr_image_dir, hr_image_dir, width, height, channel, scale, exp, repeat_count=1, precision=tf.uint8):
    lr_ds, length = raw_dataset(lr_image_dir, width, height, channel, exp, precision)
    hr_ds, _ = raw_dataset(hr_image_dir, width * scale, height * scale, channel, exp, precision)
    sr_ds, _ = raw_dataset(sr_image_dir, width * scale, height * scale, channel, exp, precision)
    ds = tf.data.Dataset.zip((lr_ds, sr_ds, hr_ds))
    ds = ds.batch(1)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

#Reference: https://github.com/krasserm/super-resolution/blob/master/data.py
#Reference: https://gist.githubusercontent.com/oldo/dc7ee7f28851922cca09/raw/3238ad3ad64eeacfcafe7c18e7e57d28b73cb007/video-metada-finder.py
class ImageDataset():
    def __init__(self, video_dir, image_dir, video_metadata, ffmpeg_option, ffmpeg_path='/usr/bin/ffmpeg', ffprobe_path='/usr/bin/ffprobe', load_on_memory=True):
        if not os.path.exists(video_dir):
            raise ValueError('directory does not exists: {}'.format(video_dir))
        if not os.path.exists(image_dir):
            raise ValueError('directory does not exists: {}'.format(image_dir))
        if not os.path.exists(ffmpeg_path):
            raise ValueError('binary does not exists: {}'.format(ffmpeg_path))
        if not os.path.exists(ffprobe_path):
            raise ValueError('binary does not exists: {}'.format(ffprobe_path))

        self.video_dir = video_dir
        self.image_dir = image_dir
        self.video_metadata = video_metadata
        self.ffmpeg_option = ffmpeg_option
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.load_on_memory = load_on_memory
        self.buffer_size = None

        self.image_datasets = {}

    @staticmethod
    def _image_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _train_patch_dataset(lr_image_dataset, hr_image_dataset, batch_size, patch_size, buffer_size, scale, load_on_memory, repeat_count=None):
        ds = tf.data.Dataset.zip((lr_image_dataset, hr_image_dataset))
        if load_on_memory: ds = ds.cache()
        #ds = tf.data.Dataset.from_tensor_slices((lr_image_dataset, hr_image_dataset))
        #ds = ds.shuffle(buffer_size=buffer_size)
        ds = ds.map(lambda lr, hr: random_crop(lr, hr, patch_size, scale), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    @staticmethod
    def _test_image_dataset(lr_image_dataset, hr_image_dataset, load_on_memory, repeat_count=1):
        ds = tf.data.Dataset.zip((lr_image_dataset, hr_image_dataset))
        #if load_on_memory: ds = ds.cache()
        #ds = tf.data.Dataset.from_tensor_slices((lr_image_dataset, hr_image_dataset))
        ds = ds.batch(1)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    @staticmethod
    def _find_video_size(video_path, ffprobe_path):
        cmd = "{} -v quiet -print_format json -show_streams".format(ffprobe_path)
        args = shlex.split(cmd)
        args.append(video_path)

        # run the ffprobe process, decode stdout into utf-8 & convert to JSON
        ffprobeOutput = subprocess.check_output(args).decode('utf-8')
        ffprobeOutput = json.loads(ffprobeOutput)

        # prints all the metadata available:
        """
        import pprint
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(ffprobeOutput)
        """

        # for example, find height and width
        height = ffprobeOutput['streams'][0]['height']
        width = ffprobeOutput['streams'][0]['width']

        return width, height

    @staticmethod
    def scale(lr, hr):
        scale = math.floor(hr/lr)
        return scale

    def _execute_ffmpeg(self, video_path, image_dir, ffmpeg_option):
        if not os.path.exists(video_path):
            raise ValueError('file does not exist: {}'.format(video_path))
        if os.path.exists(image_dir):
            logging.info('directory already exists: {}'.format(image_dir))
            return
        else:
            os.makedirs(image_dir)
            cmd = '{} -i {} {} {}/%04d.png'.format(self.ffmpeg_path, video_path, ffmpeg_option, image_dir)
            os.system(cmd)

    def _save_lr_images(self, lr, hr, scale):
        lr_video_path = os.path.join(self.video_dir, self.video_metadata.summary(lr, True))
        lr_image_dir = os.path.join(self.image_dir, '{}.{}'.format(self.video_metadata.summary(lr, True), self.ffmpeg_option.summary()))
        self._execute_ffmpeg(lr_video_path, lr_image_dir, self.ffmpeg_option.filter())

        hr_video_path = os.path.join(self.video_dir, self.video_metadata.summary(hr, False))
        lr_width, lr_height = self._find_video_size(lr_video_path, self.ffprobe_path)
        hr_width, hr_height = self._find_video_size(hr_video_path, self.ffprobe_path)

        if hr_height % lr_height == 0 and hr_width % hr_width == 0:
            return lr_image_dir, None

        target_width = int(hr_width / scale)
        target_height = int(hr_height / scale)

        lr_rescaled_image_dir = os.path.join(self.image_dir, '{}.{}'.format(self.video_metadata.summary(target_height, True), self.ffmpeg_option.summary()))
        self._execute_ffmpeg(lr_video_path, lr_rescaled_image_dir, self.ffmpeg_option.filter_rescale(target_width, target_height))

        return lr_image_dir, lr_rescaled_image_dir

    def _save_hr_images(self, hr):
        hr_video_path = os.path.join(self.video_dir, self.video_metadata.summary(hr, False))
        hr_image_dir = os.path.join(self.image_dir, '{}.{}'.format(self.video_metadata.summary(hr, False), self.ffmpeg_option.summary()))
        self._execute_ffmpeg(hr_video_path, hr_image_dir, self.ffmpeg_option.filter())
        return hr_image_dir

    def rgb_mean(self, image_dataset, image_dir):
        rgb_mean_log= os.path.join(image_dir, 'rgb_mean.log')
        if not os.path.exists(rgb_mean_log):
            images = []
            for image in image_dataset.repeat(1):
                image = tf.cast(image, tf.float32)
                images.append(image)
            images_batch = tf.stack(images, axis=0)
            images_batch_mean = tf.reduce_mean(images_batch, axis=0)
            r_mean = tf.reduce_mean(images_batch_mean[:,:,0]).numpy()
            g_mean = tf.reduce_mean(images_batch_mean[:,:,1]).numpy()
            b_mean = tf.reduce_mean(images_batch_mean[:,:,2]).numpy()

            with open(rgb_mean_log, 'wb') as f:
                f.write(struct.pack('=d', r_mean.item()))
                f.write(struct.pack('=d', g_mean.item()))
                f.write(struct.pack('=d', b_mean.item()))
        else:
            with open(rgb_mean_log, 'rb') as f:
                r_mean = struct.unpack('=d', f.read(8))[0]
                g_mean = struct.unpack('=d', f.read(8))[0]
                b_mean = struct.unpack('=d', f.read(8))[0]
        return [r_mean, g_mean, b_mean]

    def dataset(self, lr, hr, batch_size, patch_size, load_on_memory):
        scale = self.scale(lr, hr)
        if scale == 1:
            raise ValueError('scale is not valid: {}'.format(scale))

        #check if image dataset is already made
        if lr not in self.image_datasets.keys():
            lr_image_dir, lr_rescaled_image_dir = self._save_lr_images(lr, hr, scale)

            if lr_rescaled_image_dir is not None:
                lr_rescaled_image_files = sorted(glob.glob('{}/*.png'.format(lr_rescaled_image_dir)))
                lr_rescaled_image_dataset = self._image_dataset(lr_rescaled_image_files)
                rgb_mean = self.rgb_mean(lr_rescaled_image_dataset, lr_rescaled_image_dir)
                self.image_datasets[lr] = lr_rescaled_image_dataset
            else:
                lr_image_files = sorted(glob.glob('{}/*.png'.format(lr_image_dir)))
                lr_image_dataset = self._image_dataset(lr_image_files)
                rgb_mean = self.rgb_mean(lr_image_dataset, lr_image_dir)
                self.image_datasets[lr] = lr_image_dataset

        if hr not in self.image_datasets.keys():
            hr_image_dir= self._save_hr_images(hr)
            hr_image_files =  sorted(glob.glob('{}/*.png'.format(hr_image_dir)))
            hr_image_dataset = self._image_dataset(hr_image_files)
            self.image_datasets[hr] = hr_image_dataset

            if self.buffer_size is None:
                self.buffer_size = len(hr_image_files)

        #prepare Tensorflow dataset
        train_ds = self._train_patch_dataset(self.image_datasets[lr], self.image_datasets[hr], batch_size, patch_size, self.buffer_size, scale, load_on_memory)
        test_ds = self._test_image_dataset(self.image_datasets[lr], self.image_datasets[hr], load_on_memory)

        #TODO: refactor

        if lr_rescaled_image_dir is not None:
            return train_ds, test_ds, lr_rescaled_image_dir, hr_image_dir, rgb_mean, scale
        else:
            return train_ds, test_ds, lr_image_dir, hr_image_dir, rgb_mean, scale

if __name__ == '__main__':
    tf.enable_eager_execution()

    with tf.device('/cpu:0'):
        video_dir = args.video_dir
        image_dir = args.image_dir
        video_metadata = VideoMetadata(args.video_format, args.video_start_time, args.video_duration)
        ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
        ffmpeg_path = args.ffmpeg_path

        image_dataset = ImageDataset(video_dir, image_dir, video_metadata, ffmpeg_option)
        for lr, hr in args.resolution_dict.items():
            train_ds, test_ds = image_dataset.dataset(lr, hr, args.batch_size, args.patch_size, args.load_on_memory)
            it = iter(train_ds)
            for i in range(100):
                start_time = time.time()
                batch = next(it)
                end_time = time.time()
                print('1 batch elasped time: {}sec'.format(end_time - start_time))

