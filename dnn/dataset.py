import os, glob, sys, time
import numpy as np
import logging
import math
import shlex
import subprocess
import json

from PIL import Image
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.framework import tensor_shape
from tensorflow.python.data.experimental import AUTOTUNE

from option import args
from utility import FFmpegOption, VideoMetadata

#TODO: check memory usage for multiple resolutions (e.g., share target resolution frames)
#TODO: check available memory to decide load on memory or on storage

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#TODO: handle 'load_on_memory == False'
def _random_crop(lr_image, hr_image, lr_crop_size, scale):
    lr_image_shape = tf.shape(lr_image)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_image_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_image_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_crop_size = lr_crop_size * scale
    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_image_cropped = lr_image[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_image_cropped = hr_image[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_image_cropped, hr_image_cropped

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
        ds = ds.map(lambda lr, hr: _random_crop(lr, hr, patch_size, scale), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    @staticmethod
    def _test_image_dataset(lr_image_dataset, hr_image_dataset, load_on_memory, repeat_count=1):
        ds = tf.data.Dataset.zip((lr_image_dataset, hr_image_dataset))
        if load_on_memory: ds = ds.cache()
        #ds = tf.data.Dataset.from_tensor_slices((lr_image_dataset, hr_image_dataset))
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
        import pprint
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(ffprobeOutput)

        # for example, find height and width
        height = ffprobeOutput['streams'][0]['height']
        width = ffprobeOutput['streams'][0]['width']

        return width, height

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

    def dataset(self, lr, hr, batch_size, patch_size, load_on_memory):
        scale = math.floor(hr/lr)
        if scale == 1:
            raise ValueError('scale is not valid: {}'.format(scale))

        #check if image dataset is already made
        if lr not in self.image_datasets.keys():
            lr_image_dir, lr_rescaled_image_dir = self._save_lr_images(lr, hr, scale)
            if lr_rescaled_image_dir is not None:
                lr_image_files = sorted(glob.glob('{}/*.png'.format(lr_rescaled_image_dir)))
            else:
                lr_image_files = sorted(glob.glob('{}/*.png'.format(lr_image_dir)))
            lr_image_dataset = self._image_dataset(lr_image_files)
            self.image_datasets[lr] = lr_image_dataset

            if self.buffer_size is None:
                self.buffer_size = len(lr_image_files)

        if hr not in self.image_datasets.keys():
            hr_image_dir= self._save_hr_images(hr)
            hr_image_files =  sorted(glob.glob('{}/*.png'.format(hr_image_dir)))
            hr_image_dataset = self._image_dataset(hr_image_files)
            self.image_datasets[hr] = hr_image_dataset

        #prepare Tensorflow dataset
        train_ds = self._train_patch_dataset(self.image_datasets[lr], self.image_datasets[hr], batch_size, patch_size, self.buffer_size, scale, load_on_memory)
        test_ds = self._test_image_dataset(self.image_datasets[lr], self.image_datasets[hr], load_on_memory)

        return train_ds, test_ds

if __name__ == '__main__':
    tfe.enable_eager_execution()

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

